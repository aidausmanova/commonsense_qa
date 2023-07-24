import sys
import os

sys.path.insert(1, '/export/home/0usmanov/project/src/core')
import argparse
import random
import logging
import time

import numpy as np
import pandas as pd
import pyRAPL
import torch
from carbontracker.tracker import CarbonTracker
from datasets import load_dataset
from nvitop import Device
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, set_seed

def extract_questions_answers(data):
  data_rows = []

  for element in data:
    context = element['narrative']
    question = element['question']
    answer = element['answer']
    is_question_answerable = element['is_ques_answerable']
    meta = element['question_meta']

    data_rows.append({
        "question": question,
        "gold_answer": answer,
        "narrative": context,
        "meta": meta,
        "is_ques_answerable": is_question_answerable
    })
  return pd.DataFrame(data_rows)

def generate_answer(model, tokenizer, question, context):
    source_encoding = tokenizer(
        question,
        context,
        max_length=396,
        padding="max_length",
        truncation="only_second",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    generated_ids = model.generate(
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

def run(test_data, qa_model, tokenizer):
    ave_power_draw1 = 0
    ave_power_draw2 = 0
    data_rows = []
    for test_sample in tqdm(test_data):
        data_rows.append({
            "question": test_sample['question'],
            "gold_answer": test_sample['answer'],
            "narrative": test_sample['narrative'],
            "meta": test_sample['question_meta'],
            "is_ques_answerable": test_sample['is_ques_answerable'],
            "predicted_answer": generate_answer(qa_model, tokenizer, test_sample['question'], test_sample['narrative'])
        })
        power_draw = os.popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits").read().strip().split('\n')
        ave_power_draw1 += float(power_draw[0])
        ave_power_draw2 += float(power_draw[1])
    return ave_power_draw1, ave_power_draw2


if __name__ == "__main__":
    seeds = [42, 123, 456]
    root_path = "/export/home/0usmanov/project/output/"
    print("Preparing test data ...")
    tellmewhy = load_dataset('StonyBrookNLP/tellmewhy')
    test_data = tellmewhy['test']
    test_df = extract_questions_answers(tellmewhy['test'])

    print("Loading the model ...")
    model_path = "/export/home/0usmanov/project/output/tellmewhy/checkpoints/"
    # load_model = "t5s_nopretrain_Mar14_09-45-30" # t5s FT
    # load_model = "t5s_adamw_mlm_Mar12_13-48-57_Mar13_23-02-45" # t5s FT+PT
    load_model = "t5b_nopretrain_Mar14_13-46-51" # t5b FT
    # load_model = "t5b_adamw_mlm_Mar16_11-31-25_Mar17_10-59-22" # t5b FT+PT

    logging.basicConfig(filename=root_path+"inference/mylog.log", format='%(asctime)s - %(message)s', level=logging.INFO)
    # qa_model = qamodel.QAModel.load_from_checkpoint(f"{root_path}tellmewhy/checkpoints/{load_model}.ckpt")
    qa_model = T5ForConditionalGeneration.from_pretrained(model_path + load_model + '/pytorch_model.bin', local_files_only=True, config=model_path + load_model + '/config.json')
    tokenizer = T5Tokenizer.from_pretrained(model_path + load_model)
    # tokenizer = T5Tokenizer.from_pretrained('t5-base')
    qa_model.eval()

    print("Generating predictions ...")
    # for test_sample in tqdm(test_data):
    #     data_rows.append({
    #         "question": test_sample['question'],
    #         "gold_answer": test_sample['answer'],
    #         "narrative": test_sample['narrative'],
    #         "meta": test_sample['question_meta'],
    #         "is_ques_answerable": test_sample['is_ques_answerable'],
    #         "predicted_answer": generate_answer(qa_model, tokenizer, test_sample['question'], test_sample['narrative'])
    #     })
    
    for i in range(len(seeds)):
        logging.info(f"Start run for seed {seeds[i]}")
        set_seed(seeds[i])
        p1, p2 = run(test_data, qa_model, tokenizer)
        # power_draw = os.popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits").read().strip().split('\n')
        # ave_power_draw1 += float(power_draw[0])
        # ave_power_draw2 += float(power_draw[1])
        # logging.info(f"Power draw {power_draw[0]}, {power_draw[1]}")
        logging.info(f"Ave power draw {p1/len(seeds)}, {p2/len(seeds)}")
        logging.info(f"End run for seed {seeds[i]}")
    # output_df.to_csv(f"{root_path}inference/{load_model}.csv")

    print("All finished")
    