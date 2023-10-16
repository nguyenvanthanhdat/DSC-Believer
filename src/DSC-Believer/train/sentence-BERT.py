from datasets import DatasetDict
import os
from sentence_transformers import (
    losses, 
    SentenceTransformer,
    SentencesDataset
)    
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample

def create_triplet(example):
    new_example = []
    claim = example['claim']
    postitive = example['evidence']
    negative = [i for i in example['context'] if i != postitive]
    for i in range(len(negative)):
        # new_example.append(InputExample(texts=[claim, postitive, negative[i]]))
        new_example.append([claim, postitive, negative[i]])
    return {'set': new_example}


def main():
    path_root = os.getcwd()
    dataset = DatasetDict.load_from_disk(os.path.join(path_root, 'data/DSC-public-preprocess'))
    dataset = dataset.map(
        create_triplet, 
        remove_columns = ["verdict", "id", "context", "claim", "evidence"],
        # batched=True
        )
    print(dataset)
    dataset_train = dataset['train']
    n_examples = int(dataset_train.num_rows * 0.8)

    train_examples = []

    for i in range(n_examples):
        examples = dataset_train[i]['set']
        for example in examples:
            # print(example)
            train_examples.append(InputExample(texts=[example[0], example[1],example[2]]))
    model = SentenceTransformer("keepitreal/vietnamese-sbert")
    train_loss = losses.TripletLoss(model = model)
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, batch_size=4)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)


if __name__ == "__main__":
    main()
