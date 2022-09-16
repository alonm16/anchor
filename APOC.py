import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from myUtils import set_seed

class APOC:
    def __init__(self, model, tokenizer, sentences, labels, pos_tokens, neg_tokens):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.softmax = torch.nn.Softmax()
        self.pos_tokens = pos_tokens
        self.neg_tokens = neg_tokens
        
        self.pos_sentences, self.pos_shuffled_tokens, self.pos_reversed_tokens = APOC.prepare_apoc(pos_tokens, sentences, labels, 1)
        self.neg_sentences, self.neg_shuffled_tokens, self.neg_reversed_tokens = APOC.prepare_apoc(neg_tokens, sentences, labels, 0)
        
    @staticmethod
    def prepare_apoc(tokens, sentences, labels, desired_label):
        sentences = [e for i, e in enumerate(sentences) if labels[i]==desired_label]
        shuffled_tokens = tokens.copy()
        set_seed()
        random.shuffle(shuffled_tokens)
        reversed_tokens = tokens[::-1]
        
        return sentences, shuffled_tokens, reversed_tokens
    
    def _predict_scores(self, sentences):
        encoded = [[101] +[self.tokenizer.vocab[token] for token in tokens] + [102]         
                   for tokens in sentences]
        #encoded = tokenizer.encode(sentences, add_special_tokens=True, return_tensors="pt").to(device)
        to_pred = torch.tensor(encoded, device=self.device)
        outputs = self.softmax(self.model(to_pred)[0])
        return outputs.detach().cpu().numpy()

    def _remove_tokens(self, removed_tokens, sentences):
        return [['[PAD]' if token in removed_tokens else token for token in sentence] for sentence in sentences]

    def _apoc_predictions(self, sentences_arr, labels):
        predictions_arr = []
        for sentences in sentences_arr:
            predictions = [self._predict_scores([sentence])[0][label] for sentence, label in zip(sentences, labels)]
            predictions_arr.append(predictions)
        return predictions_arr

    def _apoc_formula(self, predictions_arr):
        predictions_arr = np.array(predictions_arr)
        orig_predictions = np.array(predictions_arr[0])
        N = len(orig_predictions)
        values = []

        for k in range(len(predictions_arr)):
            value = sum([sum(orig_predictions - predictions_arr[i])/N for i in range(k+1)])/(k+1)
            values.append(value) 
        return values

    def _apoc_global(self, tokens_to_remove, sentences, labels):
        tokenized_sentences = [self.tokenizer.tokenize(example) for example in sentences]
        removed_sentences_arr = [self._remove_tokens(tokens_to_remove[:i], tokenized_sentences) for i in range(len(tokens_to_remove)+1)]
        predictions_arr = self._apoc_predictions(removed_sentences_arr, labels)
        apoc_scores = self._apoc_formula(predictions_arr)
        return apoc_scores
        
    def _plot_apoc(self, scores_arr, titles, graph_title):
        plt.xlabel('# of features removed per sentence')
        plt.ylabel('APOC - global')
        
        for scores, title in zip(scores_arr, titles):
            plt.plot(range(len(scores)), scores, label = title)
        
        plt.legend()
        plt.title(graph_title)
        plt.show()
        
    def apoc_global(self):
        normal_scores = self._apoc_global(self.pos_tokens, self.pos_sentences, [1]*len(self.pos_sentences))
        random_scores = self._apoc_global(self.pos_shuffled_tokens, self.pos_sentences, [1]*len(self.pos_sentences))
        reverse_scores = self._apoc_global(self.pos_reversed_tokens, self.pos_sentences, [1]*len(self.pos_sentences))
        
        self._plot_apoc([normal_scores, random_scores, reverse_scores], ['regular', 'random', 'reverse'], 'positive')
        
        normal_scores = self._apoc_global(self.neg_tokens, self.neg_sentences, [0]*len(self.neg_sentences))
        random_scores = self._apoc_global(self.neg_shuffled_tokens, self.neg_sentences, [0]*len(self.neg_sentences))
        reverse_scores = self._apoc_global(self.neg_reversed_tokens, self.neg_sentences, [0]*len(self.neg_sentences))

        self._plot_apoc([normal_scores, random_scores, reverse_scores], ['regular', 'random', 'reverse'], 'negative')