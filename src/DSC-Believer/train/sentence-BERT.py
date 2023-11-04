from datasets import DatasetDict
import os
from sentence_transformers import (
    losses, 
    SentenceTransformer,
    SentencesDataset
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
    print('PREPARING DATA ===============================================================')
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
    # model = SentenceTransformer("Gnartiel/vietnamese-sbert", use_auth_token='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')
    model = SentenceTransformer("Gnartiel/vietnamese-sbert-v2", use_auth_token='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')
    # model = SentenceTransformer("Gnartiel/multi-sbert", use_auth_token='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')
    # model = SentenceTransformer("Gnartiel/multi-sbert-v2", use_auth_token='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')
    train_loss = losses.ContrastiveLoss(model=model)
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    num_epochs = 2
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    model_save_path = os.path.join(path_root, 'model/retrieval')
    # checkpoint_path  = os.path.join(path_root, 'model/checkpoint')
    # checkpoint_save_steps  = 200
    # checkpoint_save_total_limit = 3    
    loss_values =  model.fit(
        train_objectives=[(train_dataloader, train_loss)], 
        epochs=num_epochs,
        output_path=model_save_path,
        warmup_steps=warmup_steps,
        scheduler=scheduler,
        optimizer_class=optim.AdamW,  # Chỉ định trình tối ưu hóa mong muốn
        optimizer_params={'lr': 1e-5, 'weight_decay': 0.01},  # Tham số tối ưu hóa cụ thể
        callback=ClearMemory(),
        show_progress_bar=True)

    plt.plot(loss_values)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Plot')
    plt.savefig('training_loss.png')

    # model.save_to_hub(
    #     repo_name= "Gnartiel/multi-sbert-v2",
    #     exist_ok=True,
    # )

    model.save_to_hub(
        repo_name= "Gnartiel/vietnamese-sbert-v2",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()
