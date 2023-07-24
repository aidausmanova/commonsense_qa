import urllib

import pandas as pd
# from model.pretrain_t5.triplets_dataset import ConceptnetDataset


# def get_dataset(tokenizer, type_path, args):
#     tokenizer.max_length = args.max_seq_length
#     tokenizer.model_max_length = args.max_seq_length
#     if type_path == "train":
#       dataset = train_dataset
#     else:
#       dataset = val_dataset
#     return ConceptnetDataset(tokenizer=tokenizer, dataset=dataset)

def extract_questions_answers(data):
  data_rows = []

  for element in data:
    context = element['narrative']
    question = element['question']
    answer = element['answer']
    sentence_for_question = element['original_sentence_for_question']

    data_rows.append({
        "question": question,
        "answer": answer,
        "sentence_for_question": sentence_for_question,
        "context": context
    })
  return pd.DataFrame(data_rows)

def generate_answer(model, tokenizer, question):
  source_encoding = tokenizer(
      question["question"],
      question["context"],
      max_length=396,
      padding="max_length",
      truncation="only_second",
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

def generate_opposite_mask(model, tokenizer, sample):
  source_encoding = tokenizer(
      sample["input"],
      sample["text"],
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
  print("Len: ", len(preds))
  return "".join(preds)

def get_data(path):
  response = urllib.request.urlopen(path)
  lines = response.read().decode('utf-8')
  return lines.split('\n')


def read_data(path):
    with open(path,"r") as f:
        values = f.read().split('\n')
    return values[:-2]
