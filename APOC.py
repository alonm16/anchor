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
from score import ScoreUtils

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
        
    def _plot_apoc(self, ax, scores_arr, titles, graph_title):
        ax.set_xlabel('# of features removed')
        ax.set_ylabel('APOC - global')

        for scores, title in zip(scores_arr, titles):
            ax.plot(range(len(scores)), scores, label = title)

        ax.legend()
        ax.set_title(self.title + graph_title)
        
    def apoc_global(self, formula_type = 'v1', tokens_method = 'remove'):
        self.formula_type = formula_type
        self.tokens_method = self._remove_tokens if tokens_method=='remove' else self._replace_tokens
        legends =  ['regular', 'random', 'reverse']
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        
        
        normal_scores = self._apoc_global(self.pos_tokens, self.pos_sentences, [1]*len(self.pos_sentences))
        random_scores = np.zeros(self.num_removes+1)
        for i in range(5):
            random_scores += np.array(self._apoc_global(self.pos_shuffled_tokens[i], self.pos_sentences, [1]*len(self.pos_sentences)))
        random_scores/=5
        reverse_scores = self._apoc_global(self.pos_reversed_tokens, self.pos_sentences, [1]*len(self.pos_sentences))
        
        self._plot_apoc(axs[0], [normal_scores, random_scores, reverse_scores], legends , f' positive - {formula_type}' )
        
        normal_scores = self._apoc_global(self.neg_tokens, self.neg_sentences, [0]*len(self.neg_sentences))
        random_scores = np.zeros(self.num_removes+1)
        for i in range(5):
            random_scores += np.array(self._apoc_global(self.neg_shuffled_tokens[i], self.neg_sentences, [0]*len(self.neg_sentences)))
        random_scores/=5
        reverse_scores = self._apoc_global(self.neg_reversed_tokens, self.neg_sentences, [0]*len(self.neg_sentences))

        self._plot_apoc(axs[1], [normal_scores, random_scores, reverse_scores], legends, f' negative - {formula_type}')
        print(self.title)
        fig.savefig(f'results/graphs/{self.title}')
        
    @staticmethod
    def compare_apocs(model, tokenizer, compare_list, get_scores_fn, sentences, labels, legends, title = "", num_removes = 30, modified = False): 
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        
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
            apoc = APOC(model, tokenizer, sentences, labels, pos_tokens_arr[i], neg_tokens_arr[i], title, num_removes = num_removes, modified = modified) 
            pos_scores.append(apoc._apoc_global(apoc.pos_tokens, apoc.pos_sentences, [1]*len(apoc.pos_sentences)))
            neg_scores.append(apoc._apoc_global(apoc.neg_tokens, apoc.neg_sentences, [0]*len(apoc.neg_sentences)))
        
        # doesn't matter which apoc plots it
        apoc._plot_apoc(axs[0], pos_scores, legends, ' positive')
        apoc._plot_apoc(axs[1], neg_scores, legends, ' negative')
        
        plt.show()
        
        if modified:
            fig.savefig(f'results/graphs/{title}', bbox_inches='tight')
        else:
            fig.savefig(f'results/graphs/original/{title}', bbox_inches='tight')
        
    @staticmethod
    def compare_all(folder_name, model, tokenizer, anchor_examples, labels, title, num_removes = 30, modified = True, from_img = []):
        
        def compare_deltas():
            deltas = [0.1, 0.15, 0.2, 0.35, 0.5]
            get_scores_fn = lambda delta: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../{delta}/scores.xlsx")
            APOC.compare_apocs(model, tokenizer, deltas, get_scores_fn, anchor_examples, labels, deltas, f'{title} deltas', num_removes, modified)
        
        def compare_alphas():
            alphas = [0.95, 0.8, 0.65, 0.5]
            get_scores_fn = lambda alpha: ScoreUtils.get_scores_dict(folder_name, trail_path = "../0.1/scores.xlsx", alpha = alpha)
            APOC.compare_apocs(model, tokenizer, alphas, get_scores_fn, anchor_examples, labels, alphas, f'{title} alphas', num_removes, modified)
        
        def compare_optimizations():
            optimizations = [str(0.1), 'lossy', 'topk', 'desired']
            get_scores_fn = lambda optimization: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../{optimization}/scores.xlsx")
            APOC.compare_apocs(model, tokenizer, optimizations, get_scores_fn, anchor_examples, labels, optimizations, f'{title} optimizations', num_removes, modified)
    
        def compare_aggragations():
            aggragations = ['', '', 'sum_', 'avg_']
            alphas = [0.5, 0.95, None, None]
            legends = [0.5, 0.95, 'sum', 'avg']
            get_scores_fn = lambda x: ScoreUtils.get_scores_dict(folder_name, folder_name, trail_path = f"../0.1/{x[0]}scores.xlsx", alpha = x[1])
            APOC.compare_apocs(model, tokenizer, zip(aggragations, alphas), get_scores_fn, anchor_examples, labels, legends, f'{title} aggragations', num_removes, modified)
        
        def compare_percents():
            percents = [10, 25, 50, 75, 100]
            get_scores_fn = lambda percent: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../0.1/percents/scores-{percent}.xlsx", alpha = 0.95)
            APOC.compare_apocs(model, tokenizer, percents, get_scores_fn, anchor_examples, labels, percents, f'{title} percents', num_removes, modified)
                    
        def compare_percents_reverse():
            sorting = folder_name.split('/')[-2]
            reverse_folder = f'{folder_name}/../../{sorting}-reverse/0.1'
            percents = [10, 25, 50, 75, 100]
            get_scores_fn = lambda percent: ScoreUtils.get_scores_dict(reverse_folder, trail_path = f"../0.1/percents/scores-{percent}.xlsx", alpha = 0.95)
            APOC.compare_apocs(model, tokenizer, percents, get_scores_fn, anchor_examples, labels, percents, f'{title} percents reverse', num_removes, modified)
        
        compares = {'deltas': compare_deltas, 'alphas': compare_alphas, 'optimizations': compare_optimizations, 'aggragations': compare_aggragations, 'percents': compare_percents, 
                    'percents-reverse': compare_percents_reverse} 
        
        for c in compares:
            if c in from_img:
                plt.figure(figsize = (18,10))
                img = plt.imread(f'results/graphs/{title} {c}.png')
                imgplot = plt.imshow(img)
                plt.axis('off')
                plt.show()
            else:
                compares[c]()