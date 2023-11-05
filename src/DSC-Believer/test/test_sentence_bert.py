from datasets import DatasetDict
import os
from sentence_transformers import (
    losses, 
    SentenceTransformer,
    SentencesDataset, util
)    
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from tqdm.notebook import tqdm
import torch
from transformers.trainer_callback import TrainerCallback
import gc
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMultipleChoice, BertTokenizer, AdamW, Trainer, TrainingArguments, BertForSequenceClassification
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy
from torch import Tensor
import json

def retrieval(example, model):
  context_list = example['context']
  claim = example['claim']
  claim_embedding = model.encode(claim)
  context_embedding = model.encode(context_list)
  similar = context_list[np.argmax(util.dot_score(claim_embedding, context_embedding))]
  example['retrieval'] = similar
  return example
def evaluate(test_loader, model, loss_fn):
  model.eval()

  eval_losses = []
  with torch.no_grad():
      for eval_batch in test_loader:
          input_ids, attention_mask, labels = eval_batch
          outputs = model(input_ids.to('cuda'), attention_mask=attention_mask.to('cuda'))
          logits = outputs.logits.to('cuda')
          loss = loss_fn(logits, labels.to('cuda'))
          eval_losses.append(loss.item())

  average_eval_loss = np.average(eval_losses)
  print(f'Average Eval Loss: {average_eval_loss:.4f}')
  return average_eval_loss
    
def preprocess_data(dataset, label_encoder, tokenizer):
    inputs = tokenizer(
        dataset['claim'],
        dataset['retrieval'],
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    labels = label_encoder.fit_transform(dataset['verdict'])
    labels = torch.tensor(labels)
    return inputs, labels

class ClearMemory(TrainerCallback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        torch.cuda.empty_cache()

def main():
    path_root = os.getcwd()
    print('PREPARING DATA ===============================================================')
    dataset = DatasetDict.load_from_disk(os.path.join(path_root, 'data/DSC-public-preprocess'))
    model_name = "Gnartiel/vietnamese-sbert"
    model = SentenceTransformer(model_name).to('cuda')
    for split in dataset:
        print(f"Preprocessing {split} dataset")
        dataset[split] = dataset[split].map(retrieval, fn_kwargs={"model": model})
        
    print(dataset)
    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

    model_name = "Gnartiel/BERT-classify"
    label_encoder = preprocessing.LabelEncoder()
    model =  AutoModelForSequenceClassification.from_pretrained(model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()


    test_inputs, test_labels = preprocess_data(test_data, label_encoder, tokenizer)
     
    bs = 50 
    test_loader = DataLoader(test_dataset, batch_size=bs)
        
    best_model_state = torch.load('best_model.pth')
    model.load_state_dict(best_model_state)
    
    # Set the model to evaluation mode
    model.eval()
    predicted_labels = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            outputs = model(**inputs)
            predicted_labels.extend(outputs.logits.argmax(dim=1).cpu().numpy().tolist())
    predicted_labels = label_encoder.inverse_transform(predicted_labels)

    for idx, row in enumerate(test_data):
        row['verdict'] = predicted_labels[idx]
        if row['verdict'] == 'NEI':
            row['evidence'] = ""
    new_data = {}
    for i in test_data:
        new_data[i['id']] = {
            'verdict' : i['verdict'],
            'evidence' : i['evidence']
        }
    with open('public_result.json','w) as f:
              json.dump(new_data,f)
    

if __name__ == "__main__":
    main()
