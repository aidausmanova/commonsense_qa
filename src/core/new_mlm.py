import argparse
import ast
import gzip
import io
import json
import os
import random
import re
import tarfile
import time
import warnings
from dataclasses import dataclass, field
from io import TextIOWrapper
from typing import Any, Callable, Dict, List, NewType, Tuple, Union
from urllib.request import urlopen
import wandb

import nltk
import numpy as np
import pandas as pd
# import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from carbontracker.tracker import CarbonTracker
from comet_ml import Experiment
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from transformers import (AdamW, BatchEncoding, DataCollator,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, PreTrainedTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer, Trainer,
                          TrainerCallback, TrainerControl, TrainerState,
                          TrainingArguments, set_seed)
from transformers.optimization import Adafactor, AdafactorSchedule
from utils.services import read_data

args_dict = dict(
    train_url='https://cloud.tsinghua.edu.cn/d/670f7787b6554f308226/files/?p=%2Fcommonsense_data%2Ftrain.txt&dl=1',
    val_url='https://cloud.tsinghua.edu.cn/d/670f7787b6554f308226/files/?p=%2Fcommonsense_data%2Fvalid.txt&dl=1',
    test_url='https://cloud.tsinghua.edu.cn/d/670f7787b6554f308226/files/?p=%2Fcommonsense_data%2Ftest.txt&dl=1', 
    train_path='/export/home/0usmanov/data/train.txt',
    val_path='/export/home/0usmanov/data/val.txt',
    output_dir="/export/home/0usmanov/project/output/code_encoder/checkpoints/", 
    model_variant="t5s",
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_seq_length=512,
    learning_rate=5e-5,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=32,
    eval_batch_size=32,
    num_train_epochs=1,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=True, 
    opt_level='O1', 
    max_grad_norm=1, 
    seed=42,
)


class MaskedLMDataSet(Dataset):
    def __init__(self, sentences, mask_prob, tokenizer, max_seq_len):
        self.sentences = sentences
        self.mask_prob = mask_prob
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokenized = self.tokenizer.encode_plus(sentence, max_length=self.max_seq_len, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        masked_input_ids = input_ids.clone()
        reverse_masked_input_ids = input_ids.clone()
        mask_tokens = np.arange(32099, 31999, -1)
        reverse_mask_tokens = np.arange(32099, 31999, -1)

        for i, id in enumerate(input_ids):
            if attention_mask[i] == 1 and masked_input_ids[i] != 1 and input_ids[i] != 3:
              if random.random() < self.mask_prob:
                  # masked_input_ids[i] = tokenizer.mask_token_id
                  # Currently all masks are <extra_id_0>, but needs to increase if there is more than one masked word
                  # masked_input_ids[i] = 32099
                  masked_input_ids[i] = mask_tokens[0]
                  mask_tokens = mask_tokens[1:]
              else:
                reverse_masked_input_ids[i] = reverse_mask_tokens[0]
                reverse_mask_tokens = reverse_mask_tokens[1:]

        # return {"input_ids": masked_input_ids, "attention_mask": attention_mask, "labels": input_ids}
        return {"input_ids": masked_input_ids, "attention_mask": attention_mask, "lm_labels": reverse_masked_input_ids, "labels": reverse_masked_input_ids}


class PrinterCallback(TrainerCallback):
    def __init__(self, tracker) -> None:
       super().__init__()
       self.tracker = tracker

    def on_epoch_begin(self, args, state, control, **kwargs):
        print("EPOCH START")
        self.tracker.epoch_start()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
       print("EPOCH END")
       self.tracker.epoch_end()


if __name__ == "__main__":
    # experiment = Experiment(api_key="ggL2AArbgC6Ve3j7Ww3xMLLMK")
    root_path = "/export/home/0usmanov/project/output/"
    args = argparse.Namespace(**args_dict)
    set_seed(42)
    # pl.seed_everything(42)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("Start data loading")
    # train_lines = get_data(args.train_url)
    # radom_lines = random.choices(train_lines, k=int(len(train_lines)/50))
    # val_lines = get_data(args.val_url)

    train_lines = read_data(args.train_path)
    radom_lines = random.choices(train_lines, k=int(len(train_lines)/50))
    val_lines = read_data(args.val_path)

    print("Load model")
    tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    train_dataset = MaskedLMDataSet(train_lines, 0.15, tokenizer, 256)
    val_dataset = MaskedLMDataSet(val_lines, 0.15, tokenizer, 256)
    print("Train size: ", train_dataset.__len__())
    print("Val size: ", val_dataset.__len__())

    # Callbacks
    es_callback = EarlyStoppingCallback(early_stopping_patience=3000, early_stopping_threshold=0.0005)
    carbon_tracker = CarbonTracker(epochs=1, log_dir=root_path+"carbontracker/pretrain/")
    print_callback = PrinterCallback(carbon_tracker)

    # Specify train parameters
    # optimizer = Adafactor(model.parameters(), scale_parameter=False, 
    #                       relative_step=False, warmup_init=False,
    #                       lr=1e-4, eps=(1e-30, 1e-3),
    #                       clip_threshold=1.0)

    training_args = TrainingArguments(
        output_dir="/export/home/0usmanov/project/output/code_encoder/training_logs",
        overwrite_output_dir=True,
        num_train_epochs=1,
        learning_rate=1e-4,
        adam_epsilon=1e-8,
        optim="adamw_hf",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        weight_decay=0, 
        save_strategy="steps",
        save_steps=2000,
        evaluation_strategy="steps",
        eval_steps =2000,
        metric_for_best_model="eval_loss",
        logging_strategy="steps",
        logging_steps=2000, 
        save_total_limit=1,
        do_train = True,
        do_eval = True,
        load_best_model_at_end= True,
        # dataloader_num_workers=4,
        report_to="wandb",
        fp16=False,
    )

    # After performing tests with 10K samples for 10 epochs AdamW was faster than Adafactor for 30 mins
    # The validation loss was also lower
    trainer = Trainer(
        model=model,
        args=training_args,
        # optimizers=(optimizer, None),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[print_callback, es_callback]
    )

    print("Start training")
    training_start_time = time.time()
    local_start_time_str = time.strftime("%b%d_%H-%M-%S", time.localtime(training_start_time))

    train_result = trainer.train()
    print("Start evaluation")
    val_result = trainer.evaluate()
    model_dir = '/export/home/0usmanov/project/output/code_encoder/checkpoints/'
    trainer.save_model(model_dir + f"t5s_adamw_mlm_{local_start_time_str}")
    print("Finished")
    print("TRAIN: ", train_result)
    print("VALIDATION: ", val_result)
    
    # torch.save(model.state_dict(), model_dir+f"t5s_adafactor_mlm_{local_start_time_str}.ckpt")



    # train_result = trainer.train()
    # print("Start evaluation")
    # val_result = trainer.evaluate()
    # print("Finished")
    # print("TRAIN: ", train_result)
    # print("VALIDATION: ", val_result)
    # model_dir = '/export/home/0usmanov/project/output/code_encoder/checkpoints/'
    # trainer.save_model(model_dir + f"t5s_adafactor_mlm_{local_start_time_str}")
    # torch.save(model.state_dict(), model_dir+f"t5s_adafactor_mlm_{local_start_time_str}.ckpt")
