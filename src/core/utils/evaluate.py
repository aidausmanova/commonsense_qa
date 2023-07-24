import pandas as pd
import numpy as np
import collections
import wandb
from sentence_transformers import SentenceTransformer, util
from collections import Counter

def words_overlaps(s1, s2):
  l1 = s1.split()
  l2 = s2.split()

  multiset1 = collections.Counter(l1)
  multiset2 = collections.Counter(l2)

  overlap = list((multiset1 & multiset2).elements())
  # remainder1 = list((multiset1 - multiset2).elements())
  # remainder2 = list((multiset2 - multiset1).elements())
  overlap_percentage = (len(overlap)*100)/len(l1)

  return overlap_percentage

def get_semantic_similarity(s1, s2):
  embedding1 = model.encode(s1, convert_to_tensor=True)
  embedding2 = model.encode(s2, convert_to_tensor=True)
  cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
  return cosine_scores.item()

def get_unique_words(df, col):
  words = []
  for index, row in df.iterrows():
    words.extend(row[col].lower().split())
  return Counter(words)

def count_unique_words_vocabulary(df):
  df['clean_context'] = df['narrative'].str.replace(r'[^\w]', ' ', regex=True)
  df['clean_predicted_answer'] = df['predicted_answer'].str.replace(r'[^\w]', ' ', regex=True)

  context_words = get_unique_words(df, 'clean_context')
  answer_words = get_unique_words(df, 'clean_predicted_answer')
  set1 = set(context_words.keys())
  set2 = set(answer_words.keys())
  return set1-set2, set2-set1


if __name__ == "__main__":
    path = "/export/home/0usmanov/project/output/test_output/"
    t5_model = "t5s_no_pretrain_2023-01-16_08-47-25-v4" # t5s FT
    # t5_model = "t5s_adamw_mlm_Mar12_13-48-57_Mar13_23-02-45" # t5s FT+PT
    # t5_model = "t5b_nopretrain_Mar14_13-46-51" # t5b FT
    # t5_model = "t5b_adamw_mlm_Mar16_11-31-25_Mar17_10-59-22" # t5b FT+PT
    
    results_df = pd.read_csv(path+t5_model+".csv")

    ###############################################################################
    # Measure overlap
    # overlap_percentages = []
    # for index, row in results_df.iterrows():
    #     overlap_percentages.append(words_overlaps(row['gold_answer'], row['predicted_answer']))

    # print("Avergae overlap between answers: ", np.mean(overlap_percentages))

    ###############################################################################
    # Measure semantic similarity
    # model = SentenceTransformer('stsb-roberta-large')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    similarities = []
    my_dict = {
      "context_len": [],
      "cos_sim": [],
      "is_answerable": []
    }
    for index, row in results_df.iterrows():
      my_dict['context_len'].append(len(row['narrative']))
      my_dict['cos_sim'].append(get_semantic_similarity(row['gold_answer'], row['predicted_answer']))
      my_dict['is_answerable'].append(row['is_ques_answerable'])

    print("Mean semantic similarity score: ", np.mean(similarities))

    ###############################################################################
    # Measure vocabulary uniqueness
    # path = "/export/home/0usmanov/project/output/"
    # context_voc, answer_voc = count_unique_words_vocabulary(results_df)
    # print("Unique words in context: ", len(context_voc))
    # print("Unique words in answers: ", len(answer_voc))

    # with open(path+"unique_answer_vocabulary/"+t5_model+".txt", 'w+') as outfile:
    #   outfile.write('\n'.join(str(i) for i in answer_voc))
