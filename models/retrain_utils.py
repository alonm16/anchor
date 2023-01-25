from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pickle
import pandas as pd
import copy
from datasets import Dataset
from enum import IntEnum
import torch
import matplotlib.pyplot as plt
from utils import load_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RetrainAction(IntEnum):
    ADD = 1
    REPLACE = 2
    REMOVE = 3

class RetrainUtils:
    def __init__(self, model_name, ds_name, model_type):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
        self.ds_name = ds_name
        self.unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
        self.anchor_sentences = pickle.load(open(f"../results/retrain/{model_type}/{ds_name}/confidence/0.1/anchor_examples.pickle", "rb"))
        self.labels = pickle.load(open(f"../results/retrain/{model_type}/{ds_name}/confidence/0.1/predictions.pickle", "rb" ))
        self.model_type = model_type
        
    def get_scores_dict(self, trail_path = "scores.xlsx", alpha = 0.95):
        """
        returns dict of (anchor, score) pairs, and sum of the topk positive/negative
        """

        df = pd.read_excel(f'../results/retrain/{self.model_type}/{self.ds_name}/confidence/0.1/{trail_path}').drop(0)
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
        
    def replace_sentences(self, train_df, action = RetrainAction.ADD, top = 20):
        pos_scores, neg_scores = self.get_scores_dict(trail_path = "scores.xlsx")
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
    
    def get_tokens(self, top = 20):
        pos_scores, neg_scores = self.get_scores_dict(trail_path = "scores.xlsx")
        pos_tokens = [k for k, v in sorted(pos_scores.items(), key=lambda item: -item[1])][:top]
        neg_tokens = [k for k, v in sorted(neg_scores.items(), key=lambda item: -item[1])][:top]
        all_tokens = pos_tokens + neg_tokens
        return [self.tokenizer([t])['input_ids'][0][1] for t in all_tokens]
                
class Ensemble(torch.nn.Module):
    def __init__(self, m1, m2, threshold):
        super().__init__()
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
    
class Ensemble2(torch.nn.Module):
    def __init__(self, m1, m2):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.softmax = torch.nn.Softmax()
        
    def forward(self, x):
        outputs = self.m1(x)[0]
        scores = self.softmax(outputs)[0]
        pred_index =  torch.argmax(outputs, dim=1).cpu().numpy()[0]

        outputs2 = self.m2(x)[0]
        scores2 = self.softmax(outputs2)[0]
        pred_index2 =  torch.argmax(outputs2, dim=1).cpu().numpy()[0]
        
        if scores[pred_index]>= scores2[pred_index2]:
            return pred_index
        return pred_index2
    
        
def calc_accuracy(m, test, tokenizer, pad = False):
    """pad for gru"""
    labels = list(map(int, test['label']))
    if pad:
        sentences = [tokenizer(s, padding='max_length', max_length = 64)['input_ids'] for s in test['text']]
    else: 
        sentences = [tokenizer(s)['input_ids'] for s in test['text']]
    sentences = [torch.tensor([s], device = device) for s in sentences]
    predictions = [m(sentence) for sentence in sentences]
    return sum(p==l for l, p in zip(predictions, labels))/len(labels)

def ensemble_results(folder_name, model_name, ds_name, eval_name, validation, test, model_path = 'updated_model', orig_model_path = 'model', pad = False):
    from dataset_loader import get_ds
    def plot_ax(ax, m1, m2, thresholds, test_ds, title, legend):
        accs = []
        for th in thresholds:
            ensemble = Ensemble(m1, m2, th)
            acc = calc_accuracy(ensemble, test_ds, tokenizer, pad)[0]
            accs.append(acc)
        ax.scatter(thresholds, accs)
        ax.set_title(f'{ds_name} {title}')
        ax.legend([f'eval {legend}'])
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model(f'{folder_name}/{ds_name}/{model_path}').to(device).eval()
    orig_model = load_model(f'{folder_name}/{ds_name}/{orig_model_path}').to(device).eval()
    ths1=[0.0, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 0.96, 0.97, 0.98, 0.99, 0.993, 1]
    ths2=[0.0, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    plot_ax(axs[0,0], model, orig_model, ths1, validation, '(retrained, original)', ds_name)       
    plot_ax(axs[0,1], model, orig_model, ths1, test, '(retrained, original)', eval_name)  
    plot_ax(axs[1,0], orig_model, model, ths2, validation, '(original, retrained)', ds_name)       
    plot_ax(axs[1,1], orig_model, model, ths2, test, '(original, retrained)', eval_name)   