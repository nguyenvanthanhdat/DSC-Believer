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
    model_name = "HgThinker/vietnamese-sbert"
    model = SentenceTransformer(model_name).to('cuda')
    for split in dataset:
        print(f"Preprocessing {split} dataset")
        dataset[split] = dataset[split].map(retrieval, fn_kwargs={"model": model})
        
    print(dataset)
    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

    model_name = "vinai/phobert-base"
    label_encoder = preprocessing.LabelEncoder()
    model =  AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()


    train_inputs, train_labels = preprocess_data(train_data, label_encoder, tokenizer)
    test_inputs, test_labels = preprocess_data(test_data, label_encoder, tokenizer)
    val_inputs, val_labels = preprocess_data(val_data, label_encoder, tokenizer)
    
    train_dataset = TensorDataset(
    train_inputs.input_ids,
    train_inputs.attention_mask,
    torch.tensor(train_labels, dtype=torch.long),
    )
    
    bs = 50
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    test_dataset = TensorDataset(
        test_inputs.input_ids,
        test_inputs.attention_mask,
        torch.tensor(test_labels, dtype=torch.long),
    )
    
    test_loader = DataLoader(test_dataset, batch_size=bs)
    
    val_dataset = TensorDataset(
        val_inputs.input_ids,
        val_inputs.attention_mask,
        torch.tensor(val_labels, dtype=torch.long),
    )

    val_loader = DataLoader(val_dataset, batch_size=bs)
    
    
    num_epochs = 10
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        average_loss = []
        for batch in tqdm(train_loader):
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids.to('cuda'), attention_mask=attention_mask.to('cuda'))
            logits = outputs.logits.to('cuda')
            loss = loss_fn(logits, labels.to('cuda'))
            loss.backward()
            optimizer.step()
            average_loss.append(Tensor.cpu(loss).detach().numpy())
        epoch_loss = np.mean(average_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')
    
        eval_loss = evaluate(val_loader, model, loss_fn)
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_model = model.state_dict()
            torch.save(best_model, 'best_model.pth')
    
    
    
    best_model_state = torch.load('best_model.pth')
    model.load_state_dict(best_model_state)
    
    # Set the model to evaluation mode
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids.to('cuda'), attention_mask=attention_mask.to('cuda'))
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels.to('cuda')).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy}")
    
    model.push_to_hub(
        repo_name= "HgThinker/classify-bert",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()
