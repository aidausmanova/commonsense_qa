import random
from typing import Any, Callable, Dict, List, NewType, Tuple, Union

import torch
from new_mlm import MaskedLMDataModule, MaskedLMDataSet, MLMModel, get_data
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import (BatchEncoding, DataCollator,
                          DataCollatorForLanguageModeling, EvalPrediction,
                          HfArgumentParser, PreTrainedTokenizer, T5Config,
                          T5ForConditionalGeneration, T5Tokenizer,
                          TFAutoModelForMaskedLM, Trainer, TrainingArguments,
                          set_seed)

if __name__ == "__main__":
    # model = 't5s_model_2023-02-10_11-25-04/'
    test_url='https://cloud.tsinghua.edu.cn/d/670f7787b6554f308226/files/?p=%2Fcommonsense_data%2Ftest.txt&dl=1'
    # model_dir = '/export/home/0usmanov/project/output/code_encoder/checkpoints/t5s_new_mlm1_2023-02-21_12-55-53/'
    # model_dir = '/export/home/0usmanov/project/output/code_encoder/checkpoints/t5s_new_mlm_2023-02-22_15-44-44/'
    model_dir = '/export/home/0usmanov/project/output/code_encoder/checkpoints/t5s_tips_mlm_Mar03_13-39-57/'

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained(model_dir + 'pytorch_model.bin', local_files_only=True, config=model_dir + 'config.json')
    # model =  T5ForConditionalGeneration.from_pretrained('t5-small')
    print("T5-SMALL Parameters")
    for name, param in model.named_parameters():
        print(name)

    model.eval()

    sentences = [
        "<extra_id_0> is capable of bark",
        "dogs and <extra_id_0> are pets",
        "baseball is a <extra_id_0>",
        "<extra_id_0> eats banana",
        "students go to <extra_id_0> to study",
        "Harry and Ron are <extra_id_0>",
        "the movie was <extra_id_0>, I did not like it",
        "snakes are <extra_id_0> for human",
        "people <extra_id_0> at jokes",
        "the water in the river is <extra_id_0>",
        "<extra_id_0> is prerequisite for running"
    ]

    for sentence in sentences:
        output_ids = model.generate(tokenizer.encode(sentence, return_tensors='pt'))
        print(tokenizer.decode(output_ids[0]))

    # test_sentences = get_data(test_url)
    # dataset = MaskedLMDataSet(test_sentences[:10], 0.3, tokenizer, 128)

    # for element in dataset:
    #     # tokenized = tokenizer.encode(element, return_tensors='pt')
    #     # print(len(tokenized))
    #     # loss, output = model(tokenized['input_ids'], tokenized['attention_mask'])
    #     # output = model.generate(tokenizer.encode(element, return_tensors='pt'))
    #     sample = {
    #         'input_ids': torch.Tensor(element['input_ids'].unsqueeze(0)),
    #         'attention_mask': torch.Tensor(element['attention_mask'].unsqueeze(0)),
    #         'labels': torch.Tensor(element['labels'].unsqueeze(0))
    #     }
    #     print(type(sample), type(sample['input_ids']))

    #     output = model.generate(sample.values)
    #     print("Prediction: ", tokenizer.decode(output[0]))
