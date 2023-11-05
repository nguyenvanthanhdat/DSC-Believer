from datasets import DatasetDict
import os
from sentence_transformers import (
    losses, 
    SentenceTransformer,
    SentencesDataset
)    
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from transformers.trainer_callback import TrainerCallback
from sentence_transformers.evaluation import TripletEvaluator
from tqdm.notebook import tqdm
import gc
import torch
import torch.optim as optim
def create_triplet(example):
    new_example = []
    claim = example['claim']
    postitive = example['evidence']
    negative = [i for i in example['context'] if i != postitive]
    for i in range(len(negative)):
        # new_example.append(InputExample(texts=[claim, postitive, negative[i]]))
        new_example.append([claim, postitive, negative[i]])
    return {'set': new_example}

class ClearMemory(TrainerCallback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        torch.cuda.empty_cache()

def main():
    path_root = os.getcwd()
    dataset = DatasetDict.load_from_disk(os.path.join(path_root, 'data/DSC-public-preprocess'))
    dataset['train'] = dataset['train'].filter(lambda example: example['verdict'] != 'NEI')
    dataset = dataset.map(
        create_triplet, 
        remove_columns = ["verdict", "id", "context", "claim", "evidence"],
        # batched=True
        )
    print(dataset)
    dataset_train = dataset['train']
    max_iter = len(dataset_train)
    n_examples = int(max_iter*0.8)  
    n_examples = 3  

    train_examples = []
    for i in range(n_examples):
        examples = dataset_train[i]['set']
        for example in examples:
            # print(example)
            train_examples.append(InputExample(texts=[example[0], example[1], example[2]]))
    
    dev_examples = []
    # for i in range(n_examples, max_iter):
    for i in range(2):  
        examples = dataset_train[i]['set']
        for example in examples:
            # print(example)
            dev_examples.append(InputExample(texts=[example[0], example[1], example[2]]))
    model = SentenceTransformer("HgThinker/multi-qa-mpnet-base-dot-v1")
    train_loss = losses.TripletLoss(model = model)
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    # num_epochs = 10
    num_epochs = 2
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    scheduler = "WarmupLinear"
    model_save_path = os.path.join(path_root, 'model/retrieval')
    evaluator = TripletEvaluator.from_input_examples(dev_examples,batch_size=16,name = 'dev')
    model.fit(
        train_objectives=[(train_dataloader, train_loss)], 
        epochs=num_epochs,
        evaluation_steps=1,
        evaluator=evaluator,
        scheduler=scheduler,
        optimizer_class=optim.AdamW,  
        optimizer_params={'lr': 1e-5, 'weight_decay': 0.01},
        output_path=model_save_path,
        warmup_steps=warmup_steps,
        # callback= ClearMemory(),
        save_best_model = True,
        show_progress_bar=True
    )

    # plt.plot(loss_values)
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Plot')
    # plt.savefig('training_loss.png')
    # # model.push_to_hub('presencesw/DSC-Believer-SBERT')
    # model.save_to_hub(
    #     repo_name= "presencesw/DSC-Believer-SBERT_vTripletLoss",
    #     exist_ok=True,
    # )
    model.save_to_hub(
        repo_name= "HgThinker/multi-qa-mpnet-base-dot-v1",
        exist_ok=True,
    )
    
if __name__ == "__main__":
    main()
