from transformers import pipeline, AutoTokenizer
import pickle
import pandas as pd
import copy

class retrainUtils:
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
        max_neg = sum(df.head(top).iloc[:, list(df.columns).index(f'{alpha}-negative')+1])


        pos_keys = df[f'{alpha}-positive'].dropna().tolist()
        pos_values = df.iloc[:, list(df.columns).index(f'{alpha}-positive')+1].tolist()
        pos_scores = dict(zip(pos_keys, pos_values))
        max_pos = sum(df.head(top).iloc[:, list(df.columns).index(f'{alpha}-positive')+1])

        return pos_scores, neg_scores, max_pos, max_neg

    def _replace_tokens(self, tokens, sentences, labels):
        replaced_sentences = []
        for s, l in zip(sentences, labels):
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

        return replaced_sentences
        
    def replace_sentences(self):
        pos_scores, neg_scores, _, _ = self.get_scores_dict(trail_path = "scores.xlsx")
        pos_tokens = [k for k, v in sorted(pos_scores.items(), key=lambda item: -item[1])][:25]
        neg_tokens = [k for k, v in sorted(neg_scores.items(), key=lambda item: -item[1])][:25]
        all_tokens = pos_tokens + neg_tokens
        sentences = [self.tokenizer.tokenize(s) for s in self.anchor_sentences] 
        
        examples = self._replace_tokens(all_tokens, sentences, self.labels)
        df =  pd.DataFrame(examples, columns=['text', 'label'])
        df['label'] = df['label'].map({1: True, 0: False})
        return df