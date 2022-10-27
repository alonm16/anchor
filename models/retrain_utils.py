from transformers import pipeline, AutoTokenizer
import pickle
import pandas as pd
import copy
from datasets import Dataset
from enum import IntEnum
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RetrainAction(IntEnum):
    ADD = 1
    REPLACE = 2
    REMOVE = 3

class RetrainUtils:
    def __init__(self, model_name, ds_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
        self.ds_name = ds_name
        self.unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
        self.anchor_sentences = pickle.load(open(f"../results/{ds_name}/confidence/0.1/anchor_examples.pickle", "rb"))
        self.labels = pickle.load(open(f"../results/{ds_name}/confidence/0.1/predictions.pickle", "rb" ))
        
    def get_scores_dict(self, top=25, trail_path = "scores.xlsx", alpha = 0.95):
        """
        returns dict of (anchor, score) pairs, and sum of the topk positive/negative
        """

        df = pd.read_excel(f'../results/{self.ds_name}/confidence/0.1/{trail_path}').drop(0)
        neg_keys = df[f'{alpha}-negative'].dropna().tolist()
        neg_values = df.iloc[:, list(df.columns).index(f'{alpha}-negative')+1].tolist()
        neg_scores =dict(zip(neg_keys, neg_values))

        pos_keys = df[f'{alpha}-positive'].dropna().tolist()
        pos_values = df.iloc[:, list(df.columns).index(f'{alpha}-positive')+1].tolist()
        pos_scores = dict(zip(pos_keys, pos_values))

        return pos_scores, neg_scores

    def _replace_tokens(self, tokens, sentences, labels):
        replaced_sentences = []
        replaced_indices = []
        for idx, (s, l) in enumerate(zip(sentences, labels)):
            tokenized_sentence = copy.deepcopy(s)
            replaced = False
            for i in range(len(tokenized_sentence)):
                token = tokenized_sentence[i]
                if token in tokens:
                    replaced = True
                    tokenized_sentence[i] = '[MASK]'
                    sentence = self.tokenizer.decode(self.tokenizer.encode(tokenized_sentence))
                    results = self.unmasker(sentence, top_k=2)
                    for r in results:
                        if r['token_str']!=token:
                            tokenized_sentence[i] = r['token_str']
                            break
            if replaced:
                replaced_sentences.append([self.tokenizer.decode(self.tokenizer.encode(tokenized_sentence)[1:-1]), l])
                replaced_indices.append(idx)
                
        return replaced_sentences, replaced_indices
        
    def replace_sentences(self, train_df, action = RetrainAction.ADD):
        top = 10
        pos_scores, neg_scores = self.get_scores_dict(trail_path = "scores.xlsx", top = top)
        pos_tokens = [k for k, v in sorted(pos_scores.items(), key=lambda item: -item[1])][:top]
        neg_tokens = [k for k, v in sorted(neg_scores.items(), key=lambda item: -item[1])][:top]
        all_tokens = pos_tokens + neg_tokens
        sentences = [self.tokenizer.tokenize(s) for s in self.anchor_sentences] 
        
        examples, replaced_indices = self._replace_tokens(all_tokens, sentences, self.labels)
    
        df = pd.DataFrame(examples, columns=['text', 'label'])
        df['label'] = df['label'].map({1: True, 0: False})
        
        new_train = None
  
        if action == RetrainAction.ADD:
            new_train = pd.concat([train_df, df], ignore_index=True)
        elif action == RetrainAction.REPLACE:
            new_train = train_df.copy()
            new_train.iloc[replaced_indices] = df
        elif action == RetrainAction.REMOVE:
            new_train = train_df.drop(replaced_indices)      
        
        return Dataset.from_pandas(new_train)
    
class Ensemble(torch.nn.Module):
    def __init__(self, m1, m2, threshold):
        super(Ensemble, self).__init__()
        self.m1 = m1
        self.m2 = m2
        self.softmax = torch.nn.Softmax()
        self.threshold = threshold
        
    def forward(self, x):
        outputs = self.m1(x)[0]
        scores = self.softmax(outputs)
        if torch.max(scores, 1).values.item() > self.threshold:
            return torch.argmax(outputs, dim=1).cpu().numpy()
        outputs = self.m2(x)[0]
        return torch.argmax(outputs, dim=1).cpu().numpy()
    
def calc_accuracy(m, test, tokenizer):
    labels = list(map(int, test['label']))
    sentences = [tokenizer(s)['input_ids'] for s in test['text']]
    sentences = [torch.tensor([s], device = device) for s in sentences]
    predictions = [m(sentence) for sentence in sentences]
    return sum(p==l for l, p in zip(predictions, labels))/len(labels)