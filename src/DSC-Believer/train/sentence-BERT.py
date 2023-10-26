from datasets import DatasetDict
import os
from sentence_transformers import (
    losses, 
    SentenceTransformer,
    SentencesDataset
)    
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from tqdm.notebook import tqdm
import torch
from transformers.trainer_callback import TrainerCallback
import gc

def create_triplet(example):
    new_example = []
    claim = example['claim']
    postitive = example['evidence']
    negative = [i for i in example['context'] if i != postitive]
    new_example.append([claim, postitive, '1'])
    for i in range(len(negative)):
        # new_example.append(InputExample(texts=[claim, postitive, negative[i]]))
        new_example.append([claim, negative[i], '0'])
    return {'set': new_example}

class ClearMemory(TrainerCallback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        torch.cuda.empty_cache()

def main():
    path_root = os.getcwd()
    dataset = DatasetDict.load_from_disk(os.path.join(path_root, 'data/DSC-public-preprocess'))
    dataset['train'] = dataset['train'].filter(lambda example: example['verdict'] != 'NEI')
    print(dataset)
    
    dataset = dataset.map(
        create_triplet, 
        remove_columns = ["verdict", "id", "context", "claim", "evidence"],
        # batched=True
        )
    
    dataset_train = dataset['train']
    n_examples = len(dataset_train)    

    train_examples = []

    for i in range(n_examples):
        examples = dataset_train[i]['set']
        for example in examples:
            # print(example)
            train_examples.append(InputExample(texts=[example[0], example[1]], label=int(example[2])))
    # model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    model = SentenceTransformer("Gnartiel/vietnamese-sbert", use_auth_token='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)
    train_loss = losses.ContrastiveLoss(model=model)
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    num_epochs = 2
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    model_save_path = os.path.join(path_root, 'model/retrieval')
    # checkpoint_path  = os.path.join(path_root, 'model/checkpoint')
    # checkpoint_save_steps  = 200
    # checkpoint_save_total_limit = 3
    model.fit(
        train_objectives=[(train_dataloader, train_loss)], 
        epochs=num_epochs,
        output_path=model_save_path,
        warmup_steps=warmup_steps,
        # checkpoint_path  = checkpoint_path,
        # checkpoint_save_steps = checkpoint_save_steps,
        # checkpoint_save_total_limit = checkpoint_save_total_limit, 
        callback= ClearMemory())

    # # model.push_to_hub('presencesw/DSC-Believer-SBERT')
    # model.save_to_hub(
    #     repo_name= "presencesw/DSC-Believer-SBERT_v2",
    #     exist_ok=True,
    # )

    model.save_to_hub(
        repo_name= "Gnartiel/vietnamese-sbert",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()
