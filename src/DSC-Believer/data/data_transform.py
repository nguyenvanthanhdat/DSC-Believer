# from tqdm.notebook import tqdm
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import re
import numpy as np
import torch
import random
from sentence_transformers import SentenceTransformer, util
import os

def main():
    path_root = os.getcwd()
    dataset = DatasetDict.load_from_disk(os.path.join(path_root, 'data/DSC-public'))
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1').to('cuda')

    def split_sentences_in_context(context):
        context = context.replace(r"TP.HCM", "TP<TP>HCM")
        string = re.findall(r"[0-9A-Z]+.[0-9]+", context)
        for i in string:
            context = context.replace(i, i.replace(".","<NUMBER>"))
        list_ = re.split(r'(?<=\.)\s+', context)
        list_ = [item.replace("<NUMBER>",".") for item in list_ if item != '']
        list_ = [item.replace("<TP>",".") for item in list_]
        return list_

    def retrieval(context_list, claim):
        claim_embedding = model.encode(claim)
        context_embedding = model.encode(context_list)
        similar = context_list[np.argmax(util.dot_score(claim_embedding, context_embedding))]
        return similar

    def retrieval_top_k(context_list, claim, k):
        top_list = []
        claim_embedding = model.encode(claim)
        context_embedding = model.encode(context_list)
        full_top = util.dot_score(claim_embedding, context_embedding)
        # similar_val, similar_index = torch.topk(full_top, k)
        _, similar_index = torch.topk(full_top, k)
        for i in similar_index[0]:
            top_list.append(context_list[i])
        return top_list

    def find_similar_evi(evidence, retrieval_list):
        for i in retrieval_list:
            if i == evidence:
                return 1
        return -1

    def preprocessing(df):
        # df['context_list'] = df['context'].apply(split_sentences_in_context)
        return df
    
    for i in tqdm(dataset['train']):
        # chuyển context thành list
        i['context_list'] = split_sentences_in_context(i['context'])
        # tiến hành retrieval, trả về top 3
        i['retrieval_list'] = retrieval_top_k(i['context_list'], i['claim'], 3)
        # đếm token
        # so sánh retrieval với evidence
        # thay thế nếu không thuộc
        if i["verdict"] != "NEI":
            res = find_similar_evi(i['evidence'], i['retrieval_list'])
            if(res != -1):
                i['has_evidence'] = 1
            else:
                i['has_evidence'] = 0
                i['retrieval_list'][random.randint(0, len(i['retrieval_list']) - 1)] = i['evidence']
        else:
            i['has_evidence'] = -1
        # print(i)
        # break
    
    print('DONE TRAIN DATASET')

    for i in tqdm(dataset['dataset_public_test']):
        i['context_list'] = split_sentences_in_context(i['context'])
        i['retrieval_list'] = retrieval_top_k(i['context_list'], i['claim'], 3)

    dataset.save_to_disk(os.path.join(path_root, 'data/DSC-public-retrieval'))


if __name__ == "__main__":
    main()