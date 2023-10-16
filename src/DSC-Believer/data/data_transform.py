# from tqdm.notebook import tqdm
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import re
import numpy as np
import torch
import random
from sentence_transformers import SentenceTransformer, util
import os
from multiprocess import set_start_method
import torch

# set_start_method("spawn")
# torch.multiprocessing.set_start_method('spawn')

def split_sentences_in_context(example):
  context = example['context']
  context = context.replace(r"TP.HCM", "TP<TP>HCM")
  string = re.findall(r"[0-9A-Z]+.[0-9]+", context)
  for i in string:
    context = context.replace(i, i.replace(".","<NUMBER>"))
  list_ = re.split(r'(?<=\.)\s+', context)
  list_ = [item.replace("<NUMBER>",".") for item in list_ if item != '']
  list_ = [item.replace("<TP>",".") for item in list_]
  example['context'] = list_
  return example

def retrieval_top_k(example, k , model):
  context_list = example['context']
  claim = example['claim']
  top_list = []
  claim_embedding = model.encode(claim)
  context_embedding = model.encode(context_list)
  full_top = util.dot_score(claim_embedding, context_embedding)
  # similar_val, similar_index = torch.topk(full_top, k)
  _, similar_index = torch.topk(full_top, k)
  for i in similar_index[0]:
    top_list.append(context_list[i])
  example['retrieval'] = top_list
  return example

def find_similar_evi(example):
  retrieval_list = example['retrieval']
  evidence = example['evidence']
  verdict = example['verdict']
  if verdict == 'NEI':
    example['has_evidence'] = -1 
    return example
  for i in retrieval_list:
    if i == evidence:
      example['has_evidence'] = 1
      return example
  example['has_evidence'] = 0
  return example

def main():
    path_root = os.getcwd()
    dataset = DatasetDict.load_from_disk(os.path.join(path_root, 'data/DSC-public'))
    # model = SentenceTransformer('multi-qa-mpnet-base-dot-v1').to('cuda')

    for split in dataset:
        print(f"Preprocessing {split} dataset")
        dataset[split] = dataset[split].map(split_sentences_in_context, num_proc= 8)
    #     dataset[split] = dataset[split].map(retrieval_top_k, fn_kwargs={"k": 3, "model": model})
    
    # dataset['train'] = dataset['train'].map(find_similar_evi, num_proc= 8)

    dataset.save_to_disk(os.path.join(path_root, 'data/DSC-public-preprocess'))
    
    os.system("cd data & zip -r DSC-public-retrieval.zip DSC-public-retrieval")

if __name__ == "__main__":
    main()
