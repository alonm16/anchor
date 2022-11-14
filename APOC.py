import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter, defaultdict
from functools import reduce
from transformers import pipeline
import copy
import pandas as pd
from myUtils import set_seed

class APOC:
    def __init__(self, model, tokenizer, sentences, labels, pos_tokens, neg_tokens, title, num_removes = 25, modified = False):
        """ modified: using our modified apoc """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
        self.softmax = torch.nn.Softmax()
        self.formula_type = 'v1'
        self.tokens_method = self._remove_tokens
        self.title = title
        self.num_removes = num_removes
        self.modified = modified
        
        self.pos_sentences, self.pos_tokens, self.pos_shuffled_tokens, self.pos_reversed_tokens = self.prepare_apoc(tokenizer, pos_tokens, sentences, labels, 1)
        self.neg_sentences, self.neg_tokens, self.neg_shuffled_tokens, self.neg_reversed_tokens = self.prepare_apoc(tokenizer, neg_tokens, sentences, labels, 0)
    
    def get_tokens_arr(self, s_arr, t_arr, reverse = False):
            if self.modified:
                return [t_arr for _ in s_arr]

            sentences_tokens = []
            for sentence in s_arr:
                sentence_tokens = [t for t in t_arr if t in sentence]
                sentence_tokens.sort(key = lambda x: t_arr.index(x))
                # for non anchor words score is 0 so removal order is meaningless
                not_anchors = [w for w in sentence if w not in sentence_tokens]
                if not reverse:
                    sentence_tokens.extend(not_anchors)
                else:
                    not_anchors.extend(sentence_tokens)
                    sentence_tokens = not_anchors
                sentences_tokens.append(sentence_tokens)
            return sentences_tokens
    
    def prepare_apoc(self, tokenizer, tokens, sentences, labels, desired_label):       
        sentences = [tokenizer.tokenize(s) for i, s in enumerate(sentences) if labels[i]==desired_label]
        sentences_tokens = self.get_tokens_arr(sentences, tokens)

        shuffled_tokens_arr = []
        for i in range(1,6):
            set_seed(i*100)
            shuffled_tokens = tokens.copy()
            random.shuffle(shuffled_tokens)
            shuffled_tokens_arr.append(self.get_tokens_arr(sentences, shuffled_tokens))
        reversed_tokens = self.get_tokens_arr(sentences, tokens[::-1], reverse=True)      
        return sentences, sentences_tokens, shuffled_tokens_arr, reversed_tokens
    
    def _predict_scores(self, sentences):
        encoded = [[101] +[self.tokenizer.vocab[token] for token in tokens] + [102]         
                   for tokens in sentences]
        to_pred = torch.tensor(encoded, device=self.device)
        outputs = self.softmax(self.model(to_pred)[0])
        return outputs.detach().cpu().numpy()

    def _remove_tokens(self, idx, tokens_arr, sentences):
        return [['[PAD]' if token in tokens[:idx] else token for token in sentence] for tokens, sentence in zip(tokens_arr, sentences)]
    
    def _replace_tokens(self, idx, tokens_arr, sentences):
        replaced_sentences = []
        for tokens, s in zip(tokens_arr, sentences):
            tokenized_sentence = copy.deepcopy(s)
            for i in range(len(tokenized_sentence)):
                token = tokenized_sentence[i]
                if token in tokens[:idx]:
                    tokenized_sentence[i] = '[MASK]'
                    sentence = self.tokenizer.decode(self.tokenizer.encode(tokenized_sentence))
                    results = self.unmasker(sentence, top_k=2)
                    for r in results:
                        if r['token_str']!=token:
                            tokenized_sentence[i] = r['token_str']
                            break

            replaced_sentences.append(tokenized_sentence)
        return replaced_sentences      

    def _apoc_predictions(self, sentences_arr, labels):
        predictions_arr = []
        for sentences in sentences_arr:
            predictions = [self._predict_scores([sentence])[0][label] for sentence, label in zip(sentences, labels)]
            predictions_arr.append(predictions)
        return predictions_arr
    
    def _apoc_formula(self, orig_predictions, predictions_arr, k):
        N = len(orig_predictions)    
        return sum([sum(orig_predictions - predictions_arr[i])/N for i in range(k+1)])/(k+1)
    
    def _apoc_formula_v2(self, orig_predictions, predictions_arr, k):
        N = len(orig_predictions)
        return sum(orig_predictions - predictions_arr[k])/N 

    def _calc_apoc(self, predictions_arr):
        predictions_arr = np.array(predictions_arr)
        orig_predictions = np.array(predictions_arr[0])
        formula_fn = self._apoc_formula if self.formula_type == 'v1' else self._apoc_formula_v2
        return [formula_fn(orig_predictions, predictions_arr, k) for k in range(len(predictions_arr))]

    def _apoc_global(self, tokens, sentences, labels):
        removed_sentences_arr = []
        removed_sentences = sentences
        for i in range(self.num_removes+1):
            removed_sentences = self.tokens_method(i, tokens, removed_sentences)
            removed_sentences_arr.append(removed_sentences) 
        predictions_arr = self._apoc_predictions(removed_sentences_arr, labels)
        apoc_scores = self._calc_apoc(predictions_arr)
        return apoc_scores
        
    def _plot_apoc(self, scores_arr, titles, graph_title):
        plt.xlabel('# of features removed per sentence')
        plt.ylabel('APOC - global')
        
        for scores, title in zip(scores_arr, titles):
            plt.plot(range(len(scores)), scores, label = title)
        
        plt.legend()
        plt.title(self.title + graph_title)
        plt.show()
        
    def apoc_global(self, formula_type = 'v1', tokens_method = 'remove'):
        self.formula_type = formula_type
        self.tokens_method = self._remove_tokens if tokens_method=='remove' else self._replace_tokens
        legends =  ['regular', 'random', 'reverse']
        
        normal_scores = self._apoc_global(self.pos_tokens, self.pos_sentences, [1]*len(self.pos_sentences))
        random_scores = np.zeros(self.num_removes+1)
        for i in range(5):
            random_scores += np.array(self._apoc_global(self.pos_shuffled_tokens[i], self.pos_sentences, [1]*len(self.pos_sentences)))
        random_scores/=5
        reverse_scores = self._apoc_global(self.pos_reversed_tokens, self.pos_sentences, [1]*len(self.pos_sentences))
        
        self._plot_apoc([normal_scores, random_scores, reverse_scores], legends , f' positive - {formula_type}' )
        
        normal_scores = self._apoc_global(self.neg_tokens, self.neg_sentences, [0]*len(self.neg_sentences))
        random_scores = np.zeros(self.num_removes+1)
        for i in range(5):
            random_scores += np.array(self._apoc_global(self.neg_shuffled_tokens[i], self.neg_sentences, [0]*len(self.neg_sentences)))
        random_scores/=5
        reverse_scores = self._apoc_global(self.neg_reversed_tokens, self.neg_sentences, [0]*len(self.neg_sentences))

        self._plot_apoc([normal_scores, random_scores, reverse_scores], legends, f' negative - {formula_type}')
        
        
    @staticmethod
    def compare_apocs(model, tokenizer, compare_list, get_scores_fn, sentences, labels, legends, num_removes = 25, modified = False): 
        
        pos_tokens_arr = []
        neg_tokens_arr = []
        for item in compare_list:
            pos_scores, neg_scores = get_scores_fn(item)
            pos_tokens, neg_tokens = list(pos_scores.keys()), list(neg_scores.keys())
            pos_tokens_arr.append(pos_tokens)
            neg_tokens_arr.append(neg_tokens)
        
        pos_scores = []
        neg_scores = []
        for i in range(len(legends)):
            apoc = APOC(model, tokenizer, sentences, labels, pos_tokens_arr[i], neg_tokens_arr[i], "", num_removes = num_removes, modified = modified) 
            pos_scores.append(apoc._apoc_global(apoc.pos_tokens, apoc.pos_sentences, [1]*len(apoc.pos_sentences)))
            neg_scores.append(apoc._apoc_global(apoc.neg_tokens, apoc.neg_sentences, [0]*len(apoc.neg_sentences)))
        
        # doesn't matter which apoc plots it
        apoc._plot_apoc(pos_scores, legends, 'positive')
        apoc._plot_apoc(neg_scores, legends, 'negative')
        

class ApocUtils:
    @staticmethod
    def get_scores_dict(folder_name, top=25, trail_path = "0.1/scores.xlsx", alpha = 0.95):
        """
        returns dict of (anchor, score) pairs, and sum of the topk positive/negative
        """
        df = pd.read_excel(f'{folder_name}/{trail_path}').drop(0)

        index_prefix = f"{alpha}-" if alpha is not None else ""

        neg_keys = df[f'{index_prefix}negative'].dropna().tolist()
        neg_values = df.iloc[:, list(df.columns).index(f'{index_prefix}negative')+1].tolist()
        neg_scores =dict(zip(neg_keys, neg_values))    

        pos_keys = df[f'{index_prefix}positive'].dropna().tolist()
        pos_values = df.iloc[:, list(df.columns).index(f'{index_prefix}positive')+1].tolist()
        pos_scores = dict(zip(pos_keys, pos_values))

        return pos_scores, neg_scores

    # get all anchor above 0.95, multiple in a sentence
    @staticmethod
    def get_best(explanations):
        best_exps = dict()
        for exp in explanations:
            if exp.precision < 0.95:
                continue
            if exp.index not in best_exps.keys():
                best_exps[exp.index]=[exp]
            else:
                best_exps[exp.index].append(exp)
        print(len(best_exps))
        return reduce(lambda x,y: x+y, best_exps.values())
    
    @staticmethod
    def get_anchor_occurences(explanations):
        c = Counter()
        for exp in explanations:
            c.update([exp.names[0]])

        return c
    
    @staticmethod
    def get_normal_occurences(sentences, anchor_occurences, tokenizer):
        c = Counter()
        for sentence in sentences:
            c.update(tokenizer.tokenize(sentence))

        #removing occurences of the words as anchor
        for word in anchor_occurences.keys():
            c[word]-=anchor_occurences[word]

        return c

    @staticmethod
    def calculate_sum(anchor_occurences, normal_occurences):
        sums = dict()
        sum_occurences = sum(anchor_occurences.values())
        for word, count in anchor_occurences.items():
            sums[word] = count/sum_occurences

        return sums

    @staticmethod
    def calculate_avg(anchor_occurences, normal_occurences):
        avgs = dict()
        for word, count in anchor_occurences.items():
            avgs[word] = count/(anchor_occurences[word]+normal_occurences[word])

        return avgs
    
    @staticmethod
    def calculate_score(folder_name, tokenizer, anchor_examples, explanations, labels, agg_name):
        aggs = {'sum': ApocUtils.calculate_sum, 'avg': ApocUtils.calculate_avg}
        columns = ['name', 'anchor score', 'type occurences', 'total occurences','+%', '-%', 'both', 'normal']

        exps = ApocUtils.get_best(explanations)
        pos_exps = [exp for exp in exps if labels[exp.index]==0]
        neg_exps = [exp for exp in exps if labels[exp.index]==1]

        anchor_occurences = ApocUtils.get_anchor_occurences(exps)
        pos_occurences = ApocUtils.get_anchor_occurences(pos_exps)
        neg_occurences = ApocUtils.get_anchor_occurences(neg_exps)

        normal_occurences = ApocUtils.get_normal_occurences(anchor_examples, anchor_occurences, tokenizer)
        df_pos, df_neg = [], []

        teta_pos = aggs[agg_name](pos_occurences, normal_occurences)
        teta_neg = aggs[agg_name](neg_occurences, normal_occurences)

        for anchor, score in teta_pos.items():
            pos_percent = round((pos_occurences[anchor])/anchor_occurences[anchor], 2)
            neg_percent = 1-pos_percent
            both = pos_occurences[anchor]>0 and neg_occurences[anchor]>0
            df_pos.append([anchor, score , pos_occurences[anchor], anchor_occurences[anchor], pos_percent, neg_percent, both,  normal_occurences[anchor]]) 


        for anchor, score in teta_neg.items():
            pos_percent = round((pos_occurences[anchor])/anchor_occurences[anchor], 2)
            neg_percent = 1-pos_percent
            both = pos_occurences[anchor]>0 and neg_occurences[anchor]>0
            df_neg.append([anchor, score, neg_occurences[anchor], anchor_occurences[anchor], pos_percent, neg_percent, both,  normal_occurences[anchor]])

        df_pos.sort(key=lambda exp: -exp[1])
        df_neg.sort(key=lambda exp: -exp[1])
        df_pos = pd.DataFrame(data = df_pos, columns = columns ).set_index('name')
        df_neg = pd.DataFrame(data = df_neg, columns = columns ).set_index('name')

        writer = pd.ExcelWriter(f'{folder_name}/{agg_name}_scores.xlsx',engine='xlsxwriter') 

        workbook=writer.book
        worksheet=workbook.add_worksheet('Sheet1')
        writer.sheets['Sheet1'] = worksheet

        cur_col = 0
        is_positive = False

        for df in [df_pos, df_neg]:
            cur_type = 'positive' if is_positive else 'negative'
            is_positive = not is_positive
            worksheet.write(0, cur_col, f'{cur_type}')
            df.to_excel(writer, sheet_name=f'Sheet1', startrow=1, startcol=cur_col)
            cur_col+= len(columns) + 1

        writer.save()