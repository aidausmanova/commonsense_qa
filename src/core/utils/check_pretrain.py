import sys
sys.path.insert(1, '/export/home/0usmanov/project/src/core')
import random
import argparse
import torch

from carbontracker.tracker import CarbonTracker
from datasets import load_dataset
from nvitop import Device
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.services import extract_questions_answers, generate_opposite_mask
from utils.constants import PRETRAIN_PARAMETERS, FINETUNE_PARAMETERS
# import pretrain.T5Pretrainer as trainer_model

from pretrain import T5Pretrainer


def generate_answer(model, tokenizer, input):
    input_ = input.lower() + ' </s>'

    source_encoding = tokenizer(
        [input_],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    generated_ids = model.model.generate(
        input_ids=source_encoding["input_ids"],
        attention_mask=source_encoding["attention_mask"],
        num_beams=1,
        max_length=80,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    preds = [
        tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    ]

    return "".join(preds)

if __name__ == "__main__":
    t5_model = "t5-small"
    root_path = "/export/home/0usmanov/project/output/"
    load_model = "t5s_weighted_pretrainer_model_2023-01-05_08-56-25-v4"
    tokenizer = T5Tokenizer.from_pretrained(t5_model)

    print("Preparing test data ...")
    path = "/export/home/0usmanov/data/conceptnet_df_w/test.csv"
    test_data = pd.read_csv(path)

    data_rows = []

    print("Loading the model ...")
    trainer_model = T5Pretrainer.load_from_checkpoint(f"{root_path}conceptnet/checkpoints/{load_model}.ckpt")
    trainer_model.freeze()

    print("Generating predictions ...")
    for ind, row in tqdm(test_data.iterrows()):
        data_rows.append({
            "target": row['target'],
            "predicted_answer": generate_answer(trainer_model, tokenizer, row['input']),
            "text": row['text']
        })

    
    finish_time = time.time()
    local_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(finish_time))
    print(f"Finished at {local_time_str}")
    output_df = pd.DataFrame(data_rows)
    output_df.to_csv(f"{root_path}test_output/{load_model}_{local_time_str}.csv")
