import sys

sys.path.insert(1, '/export/home/0usmanov/project/src/core')
import argparse
import random
import time
from datetime import timedelta
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
# import pytorch_lightning as pl
import torch
from carbontracker import parser
from carbontracker.tracker import CarbonTracker
from comet_ml import Experiment
from datasets import load_dataset
# from model.qa.tmw_dataset import TellMeWhyDataset
from nvitop import Device
# from pytorch_lightning.callbacks import Checkpoint, ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.loops import FitLoop
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, EarlyStoppingCallback,
                          T5ForConditionalGeneration, T5Tokenizer, Trainer,
                          TrainerCallback, TrainerControl, TrainerState,
                          TrainingArguments, set_seed)
# from lightning_fabric.utilities.types import _PATH
from transformers.optimization import Adafactor, AdafactorSchedule
from utils.constants import FINETUNE_PARAMETERS, PRETRAIN_PARAMETERS
from utils.services import extract_questions_answers, generate_opposite_mask

args_dict = dict(
    root_path = "/export/home/0usmanov/project/output/",
    output_dir="", # path to save the checkpoints
    model_variant="t5s",
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_seq_length=512,
    learning_rate=5e-5,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=16,
    eval_batch_size=16,
    num_train_epochs=50,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
)

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer


class TellMeWhyDataset(Dataset):
  def __init__(
      self,
      data: pd.DataFrame,
      tokenizer: T5Tokenizer,
      source_max_token_len: int = 75,
      target_max_token_len: int = 30
      # source_max_token_len: int = 396,
      # target_max_token_len: int = 32
  ):

    self.tokenizer = tokenizer
    self.data = data
    self.source_max_token_len = source_max_token_len
    self.target_max_token_len = target_max_token_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    source_encoding = self.tokenizer(
        data_row["question"],
        data_row["context"],
        max_length=self.source_max_token_len,
        padding="max_length",
        truncation="only_second",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    target_encoding = self.tokenizer(
        data_row["answer"],
        max_length=self.target_max_token_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    labels = target_encoding["input_ids"]
    labels[labels==0] = -100

    return dict(
        question=data_row["question"],
        context=data_row["context"],
        answer=data_row["answer"],
        input_ids=source_encoding["input_ids"].flatten(),
        attention_mask=source_encoding["attention_mask"].flatten(),
        labels=labels.flatten()
    )


class PrinterCallback(TrainerCallback):
    def __init__(self, tracker) -> None:
       super().__init__()
       self.tracker = tracker

    # def on_train_begin(self, args, state, control, **kwargs):
        # print('\033[1m'+ '=' * 25 + " Model Training " + '=' * 25 + '\033[0m')

    def on_epoch_begin(self, args, state, control, **kwargs):
        # print('\n'+ '\033[1m'+ '=' * 25 +' Epoch {:} / {:} '.format(int(trainer.state.epoch) + 1, int(trainer.state.num_train_epochs)) + '=' * 25)
        print("EPOCH START")
        self.tracker.epoch_start()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
       print("EPOCH END")
       self.tracker.epoch_end()

if __name__ == "__main__":
    set_seed(42)
    args = argparse.Namespace(**args_dict)
    # experiment = Experiment(api_key="ggL2AArbgC6Ve3j7Ww3xMLLMK")
    
    root_path = "/export/home/0usmanov/project/output/"
    model_path = "/export/home/0usmanov/project/output/code_encoder/checkpoints/"
    # model_name = "t5s_new_mlm_2023-02-22_15-44-44"

    model_name = "t5s_adamw_mlm_Mar12_13-48-57"
    # model_name = "t5b_adamw_mlm_Mar16_11-31-25"
    # model_name = "t5s_nopretrain"

    print("Loading dataset ...")
    tellmewhy = load_dataset('StonyBrookNLP/tellmewhy')
    train_df = extract_questions_answers(tellmewhy["train"])
    train_df = train_df.sample(frac = 1)
    val_df = extract_questions_answers(tellmewhy["validation"])

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    train_dataset = TellMeWhyDataset(train_df, tokenizer, 256, 128)
    val_dataset = TellMeWhyDataset(val_df, tokenizer, 256, 128)

    print("Creating QA model ...")
    # model_path = "/export/home/0usmanov/project/output/code_encoder/training_logs/checkpoint-170000/"
    qa_model = T5ForConditionalGeneration.from_pretrained(model_path + model_name + '/pytorch_model.bin', 
                                                          local_files_only=True, 
                                                          config=model_path + model_name + '/config.json')
    # qa_model = T5ForConditionalGeneration.from_pretrained('t5-small')    

    # Define callbacks
    es_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0005)
    # comet_callback = CometCallback(COMET_PROJECT_NAME="t5_finetuning")
    carbon_tracker = CarbonTracker(epochs=40, log_dir=root_path+"carbontracker/finetune/")
    print_callback = PrinterCallback(carbon_tracker)
    
    training_start_time = time.time()
    local_start_time_str = time.strftime("%b%d_%H-%M-%S", time.localtime(training_start_time))
    print("Start training at " + local_start_time_str)

    # Specify train parameters
    # NOTE: T5 authors recommend using Adafactor
    # optimizer = Adafactor(qa_model.parameters(), scale_parameter=False, 
    #                       relative_step=False, warmup_init=False, 
    #                       lr=0.001, eps=(1e-30, 1e-3),
    #                       clip_threshold=1.0, weight_decay=0.0)

    training_args = TrainingArguments(
        output_dir="/export/home/0usmanov/project/output/tellmewhy/training_logs",
        overwrite_output_dir=True,
        # num_train_epochs=args.num_train_epochs,
        num_train_epochs=40,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        fp16=False,
        weight_decay=0, 
        save_steps=500,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        # eval_steps = 500,
        metric_for_best_model="eval_loss",
        logging_strategy="epoch",
        # logging_steps=500, 
        save_total_limit=1,
        do_train = True,
        do_eval = True,
        load_best_model_at_end= True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=qa_model,
        args=training_args,
        # optimizers=(optimizer, None),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[es_callback, print_callback]
    )

    print("Start training")
    trainer.train()
    print("Start evaluation")
    trainer.evaluate()
    carbon_tracker.stop()
    print("Finished")
    save_dir = '/export/home/0usmanov/project/output/tellmewhy/checkpoints/'

    # model_name = "t5s_adamw_mlm_Mar12_13-48-57"
    # model_name = "t5l_nopretrain"
    trainer.save_model(save_dir + f"{model_name}_{local_start_time_str}")
    # torch.save(qa_model.state_dict(), save_dir+f"{model_name}_bs16_{local_start_time_str}.ckpt")

    # trainer.save_model(save_dir + f"nopretrain_{local_start_time_str}")
    # torch.save(qa_model.state_dict(), save_dir + f"nopretrain_{local_start_time_str}.ckpt")
