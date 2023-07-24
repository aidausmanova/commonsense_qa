import tarfile
import gzip
import re
import io
import json
import ast
from urllib.request import urlopen
import urllib
import pandas as pd
import random
from io import TextIOWrapper
import random

from constants import RELATION_TEMPLATES

def removeLastN(S, N):
    S = S[:len(S)-N]
    return S


def get_data(path):
  response = urllib.request.urlopen(path)
  lines = TextIOWrapper(response, encoding='utf-8')
  return pd.DataFrame({'text':lines})


def parse_assertions(save_dict, triples):
    pattern = re.compile(r"^\/a\/\[.*\/,\/c\/en\/.*\/,\/c\/en\/")

    with gzip.open('drive/MyDrive/Study/Masters/Thesis/conceptnet-assertions-5.7.0.csv.gz','r') as tar:
        for line in tar:     
            string = line.decode('utf-8')
            if pattern.search(string): 
                my_str = string.split("\t")
                if 'surfaceText' in my_str[4]:
                    # if len(save_dict) < 10000:
                    my_json = json.loads(my_str[4])

                    if triples:
                        if my_json['surfaceStart'] not in save_dict:
                            save_dict[my_json['surfaceStart']] = RELATION_TEMPLATES[my_json['relation']].format(my_json['surfaceStart'],  my_json['surfaceEnd'])
                        else:
                            save_dict[my_json['surfaceStart']].append(RELATION_TEMPLATES[my_json['relation']].format(my_json['surfaceStart'],  my_json['surfaceEnd']))
                    else:
                        if my_json['surfaceStart'] not in save_dict:
                            save_dict[my_json['surfaceStart']] = [my_json['surfaceText'].replace('[','').replace(']','')]
                        else:
                            save_dict[my_json['surfaceStart']].append(my_json['surfaceText'].replace('[','').replace(']',''))


def sample_triples(file, triples):
    with open('data\en-conceptnet_surfacetex.json') as json_file:
        data = json.load(json_file)

    # randomly sample 10 relations for each subject
    assertions_str = []
    for key, arr in data.items():
        sub_str = ""
        if triples:
            for tuplee in (random.sample(arr, 10) if (len(arr) > 10) else arr):
                sub_str += key + " | "+tuplee[0]+" | "+tuplee[1]+" && "
        else:
            for sentence in (random.sample(arr, 5) if (len(arr) > 5) else arr):
            sub_str += sentence.replace('[','').replace(']','')+" && "

        file.write(str(removeLastN(sub_str, 4))+"\n")


if __name__ == "__main__":
    save_dict = {}  
    triples = False

    parse_assertions(save_dict, triples)

    f = open("en_10_triples_string.txt", "w")
    sample_triples(triples)
