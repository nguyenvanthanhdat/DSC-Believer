from tqdm.notebook import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from datasets import DatasetDict
import os

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0].to('cuda')  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float().to('cuda')
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    path_root = os.getcwd()
    dataset = DatasetDict.load_from_disk(os.path.join(path_root, 'data/DSC-public-preprocess'))
    model_name = "Gnartiel/vietnamese-sbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, use_auth_token='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').to('cuda')

    dataset_test = dataset['validation'].filter(lambda example: example['verdict'] != 'NEI')
    print(dataset_test)

    n_samples_test = len(dataset_test)
    predict_score = 0
    for i in tqdm(range(n_samples_test)):
        claim = dataset_test[i]['claim']
        context = dataset_test[i]['context']
        evidence = dataset_test[i]['evidence']

        encoded_input = tokenizer([claim] + context, padding=True, truncation=True, return_tensors="pt").to('cuda')
        with torch.no_grad():
            output = model(**encoded_input)

        sentence_embeddings = mean_pooling(output, encoded_input['attention_mask'])

        claim_embedding = sentence_embeddings[0].unsqueeze(0)  # Extract claim embedding
        context_embeddings = sentence_embeddings[1:]  # Extract context embeddings
        cosine_scores = F.cosine_similarity(claim_embedding, context_embeddings)

        max_score_index = torch.argmax(cosine_scores).item()
        max_score_sentence = context[max_score_index]
        if max_score_sentence == evidence:
            predict_score += 1
    
    print(f"Performance of model with validation dataset: ", predict_score/n_samples_test*100, "%")

if __name__ == "__main__":
    main()
