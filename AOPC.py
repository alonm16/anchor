import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter, defaultdict
from functools import reduce
from transformers import pipeline
import copy
import pandas as pd
import seaborn as sns
from myUtils import set_seed
from score import ScoreUtils

class AOPC_Plotter:
    @staticmethod
    def aopc_plot(pos_scores_arr, neg_scores_arr, legends, graph_title):
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        fig.supxlabel('# of features removed', x=0.51)
        fig.supylabel('AOPC - global', x=0.08)
        fig.suptitle(graph_title)
        
        arrs = [pos_scores_arr, neg_scores_arr]
        ax_titles = ['positive', 'negative']
        
        for i in range(len(axs)):
            for scores, legend in zip(arrs[i], legends):
                axs[i].plot(range(len(scores)), scores, label = legend)
            axs[i].legend()
            axs[i].set_title(ax_titles[i])
            
        fig.savefig(f'results/graphs/{graph_title}', bbox_inches='tight')
        
    @staticmethod       
    def time_aopc_plot(pos_scores_arr, neg_scores_arr, legends, graph_title):
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        fig.supxlabel('percents of dataset', x=0.51)
        fig.supylabel('AOPC - global', x=0.08)
        fig.suptitle(graph_title)
        
        arrs = [np.array(pos_scores_arr), np.array(neg_scores_arr)]
        ax_titles = ['positive', 'negative']
        aopc_removes = [1, 5, 10, 20, 30]
        percents = [10, 25, 50, 75, 100]
        
        for i in range(len(axs)):
            time_scores_arr = np.transpose(arrs[i])
            time_scores_arr = time_scores_arr[aopc_removes, :]

            for scores, legend in zip(time_scores_arr, aopc_removes):
                axs[i].plot(percents, scores, label = legend)
            axs[i].legend()
            axs[i].set_title(ax_titles[i])
            
        fig.savefig(f'results/graphs/{graph_title}', bbox_inches='tight')

class AOPC:
    def __init__(self, model, tokenizer, sentences, labels, pos_tokens, neg_tokens, title, num_removes = 25, modified = False):
        """ modified: using our modified aopc """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
        self.softmax = torch.nn.Softmax()
        self.tokens_method = self._remove_tokens
        self.title = title
        self.num_removes = num_removes
        self.modified = modified
        
        self.pos_sentences, self.pos_tokens, self.pos_shuffled_tokens, self.pos_reversed_tokens = self.prepare_aopc(tokenizer, pos_tokens, sentences, labels, 1)
        self.neg_sentences, self.neg_tokens, self.neg_shuffled_tokens, self.neg_reversed_tokens = self.prepare_aopc(tokenizer, neg_tokens, sentences, labels, 0)
    
    def get_tokens_arr(self, s_arr, t_arr, reverse = False):
            if self.modified:
                return [t_arr]*len(s_arr)

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
    
    def prepare_aopc(self, tokenizer, tokens, sentences, labels, desired_label):       
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

    def _predict_scores(self, sentences):
        pad = max(len(s) for s in sentences)
        input_ids = [[101] +[self.tokenizer.vocab[token] for token in tokens] + [102] + [0]*(pad-len(tokens)) for tokens in sentences]
        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = [[1]*(len(tokens)+2)+[0]*(pad-len(tokens)) for tokens in sentences]
        attention_mask = torch.tensor(attention_mask, device=self.device)
        outputs = self.softmax(self.model(input_ids = input_ids, attention_mask = attention_mask)[0])
        return outputs.detach().cpu().numpy()
    
    def _aopc_predictions(self, sentences_arr, label):
        return [self._predict_scores(sentences)[:, label] for sentences in sentences_arr]
    
    def _aopc_formula(self, orig_predictions, predictions_arr, k):
        N = len(orig_predictions)    
        return sum([sum(orig_predictions - predictions_arr[i])/N for i in range(k+1)])/(k+1)

    def _calc_aopc(self, predictions_arr):
        predictions_arr = np.array(predictions_arr)
        orig_predictions = predictions_arr[0]
        return [self._aopc_formula(orig_predictions, predictions_arr, k) for k in range(len(predictions_arr))]

    def _aopc_global(self, tokens, sentences, label):
        removed_sentences_arr = []
        removed_sentences = sentences
        for i in range(self.num_removes+1):
            removed_sentences = self.tokens_method(i, tokens, removed_sentences)
            removed_sentences_arr.append(removed_sentences) 
        predictions_arr = self._aopc_predictions(removed_sentences_arr, label)
        aopc_scores = self._calc_aopc(predictions_arr)
        return aopc_scores  
        
    def aopc_global(self, tokens_method = 'remove', legends = ['normal', 'random', 'reverse'], plotter = AOPC_Plotter.aopc_plot):
        self.tokens_method = self._remove_tokens if tokens_method=='remove' else self._replace_tokens
        pos_scores = []
        if 'normal' in legends:
            pos_scores.append(self._aopc_global(self.pos_tokens, self.pos_sentences, 1))
        
        if 'random' in legends:
            random_scores = np.zeros(self.num_removes+1)
            for i in range(5):
                random_scores += np.array(self._aopc_global(self.pos_shuffled_tokens[i], self.pos_sentences, 1))
            random_scores/=5
            pos_scores.append(random_scores)
        
        if 'reverse' in legends:
            pos_scores.append(self._aopc_global(self.pos_reversed_tokens, self.pos_sentences, 1))
                
        neg_scores = []
        if 'normal' in legends:
            neg_scores.append(self._aopc_global(self.neg_tokens, self.neg_sentences, 0))
        
        if 'random' in legends:
            random_scores = np.zeros(self.num_removes+1)
            for i in range(5):
                random_scores += np.array(self._aopc_global(self.neg_shuffled_tokens[i], self.neg_sentences, 0))
            random_scores/=5
            neg_scores.append(random_scores)
            
        if 'reverse' in legends:
            neg_scores.append(self._aopc_global(self.neg_reversed_tokens, self.neg_sentences, 0))

        plotter(pos_scores, neg_scores, legends, self.title)
        
    @staticmethod
    def compare_aopcs(model, tokenizer, compare_list, get_scores_fn, sentences, labels, legends, title = "", num_removes = 30, modified = False, plotter = AOPC_Plotter.aopc_plot):         
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
            aopc = AOPC(model, tokenizer, sentences, labels, pos_tokens_arr[i], neg_tokens_arr[i], title, num_removes = num_removes, modified = modified) 
            pos_scores.append(aopc._aopc_global(aopc.pos_tokens, aopc.pos_sentences, 1))
            neg_scores.append(aopc._aopc_global(aopc.neg_tokens, aopc.neg_sentences, 0))
        
        plotter(pos_scores, neg_scores, legends, title)
        
    @staticmethod
    def compare_all(folder_name, model, tokenizer, anchor_examples, labels, title, num_removes = 30, modified = True, from_img = []):
        
        def compare_deltas():
            deltas = [0.1, 0.15, 0.2, 0.35, 0.5]
            get_scores_fn = lambda delta: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../{delta}/scores.xlsx")
            AOPC.compare_aopcs(model, tokenizer, deltas, get_scores_fn, anchor_examples, labels, deltas, f'{title} deltas', num_removes, modified)
        
        def compare_alphas():
            alphas = [0.95, 0.8, 0.65, 0.5]
            get_scores_fn = lambda alpha: ScoreUtils.get_scores_dict(folder_name, trail_path = "../0.1/scores.xlsx", alpha = alpha)
            AOPC.compare_aopcs(model, tokenizer, alphas, get_scores_fn, anchor_examples, labels, alphas, f'{title} alphas', num_removes, modified)
        
        def compare_optimizations():
            optimizations = [str(0.1), 'lossy', 'topk', 'desired']
            get_scores_fn = lambda optimization: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../{optimization}/scores.xlsx")
            AOPC.compare_aopcs(model, tokenizer, optimizations, get_scores_fn, anchor_examples, labels, optimizations, f'{title} optimizations', num_removes, modified)
    
        def compare_aggragations():
            aggragations = ['', '', 'sum_', 'avg_']
            alphas = [0.5, 0.95, None, None]
            legends = ['probabilistic α=0.5', 'probabilistic α=0.95', 'sum', 'avg']
            get_scores_fn = lambda x: ScoreUtils.get_scores_dict(folder_name, folder_name, trail_path = f"../0.1/{x[0]}scores.xlsx", alpha = x[1])
            AOPC.compare_aopcs(model, tokenizer, zip(aggragations, alphas), get_scores_fn, anchor_examples, labels, legends, f'{title} aggragations', num_removes, modified)
        
        def compare_percents():
            percents = [10, 25, 50, 75, 100]
            get_scores_fn = lambda percent: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../0.1/percents/scores-{percent}.xlsx", alpha = 0.95)
            AOPC.compare_aopcs(model, tokenizer, percents, get_scores_fn, anchor_examples, labels, percents, f'{title} percents', num_removes, modified)
        
        def compare_time_percents():
            percents = [10, 25, 50, 75, 100]
            get_scores_fn = lambda percent: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../0.1/percents/scores-{percent}.xlsx", alpha = 0.95)
            AOPC.compare_aopcs(model, tokenizer, percents, get_scores_fn, anchor_examples, labels, percents, f'{title} time-percents', num_removes, modified, plotter=AOPC_Plotter.time_aopc_plot)
            
        compares = {'deltas': compare_deltas, 'alphas': compare_alphas, 'optimizations': compare_optimizations, 'aggragations': compare_aggragations, 'percents': compare_percents, 'time-percents': compare_time_percents} 

        for c in compares:
            if c in from_img:
                plt.figure(figsize = (15, 5))
                img = plt.imread(f'results/graphs/{title} {c}.png')
                imgplot = plt.imshow(img)
                plt.axis('off')
                plt.show()
            else:
                compares[c]()
                
    @staticmethod
    def compare_random_aopcs(folder_name, model, tokenizer, seeds, compare_list, sentences, labels, legends, title = "", num_removes = 30, modified = False,): 
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        
        def random_helper(get_score_fn, pos_df, neg_df):
            pos_tokens_arr = []
            neg_tokens_arr = []
            for item in compare_list:
                pos_scores, neg_scores = get_scores_fn(item)
                pos_tokens, neg_tokens = list(pos_scores.keys()), list(neg_scores.keys())
                pos_tokens_arr.append(pos_tokens)
                neg_tokens_arr.append(neg_tokens)

            for i in range(len(legends)):
                aopc = AOPC(model, tokenizer, sentences, labels, pos_tokens_arr[i], neg_tokens_arr[i], title, num_removes = num_removes, modified = modified) 
                pos_result = aopc._aopc_global(aopc.pos_tokens, aopc.pos_sentences, 1)

                neg_result = aopc._aopc_global(aopc.neg_tokens, aopc.neg_sentences, 0)
            
                pos_df = pos_df.append(pd.DataFrame({"# of removed features": np.arange(num_removes+1), "AOPC global": pos_result, "delta": np.repeat(legends[i], num_removes+1)}))
                neg_df = neg_df.append(pd.DataFrame({"# of removed features": np.arange(num_removes+1), "AOPC global": neg_result, "delta": np.repeat(legends[i], num_removes+1)}))
            
            return pos_df, neg_df
        
        pos_df = pd.DataFrame(columns = ["# of removed features", "AOPC global", "delta"])
        neg_df = pd.DataFrame(columns = ["# of removed features", "AOPC global", "delta"])
        for i, seed in enumerate(seeds):
            seed_folder = f'{folder_name}/../seed/{seed}/0.1'
            get_scores_fn = lambda delta: ScoreUtils.get_scores_dict(seed_folder, trail_path = f"../{delta}/scores.xlsx")
            pos_df, neg_df = random_helper(get_scores_fn, pos_df, neg_df)

        sns.lineplot(data=pos_df, x="# of removed features", y="AOPC global", hue = "delta", legend = False, ax = axs[0], palette=sns.color_palette())
        #sns.boxplot(data=pos_df, x="# of removed features", y="AOPC global", hue = "delta", ax = axs[0]).set(title = title+' positive')
        sns.lineplot(data=neg_df, x="# of removed features", y="AOPC global", hue = "delta", legend = False, ax = axs[1], palette=sns.color_palette())
        #sns.boxplot(data=neg_df, x="# of removed features", y="AOPC global", hue = "delta", ax = axs[1]).set(title = title + ' negative')
                                                                                        
        axs[0].set_xticks(np.arange(0, num_removes, 5)) 
        axs[1].set_xticks(np.arange(0, num_removes, 5)) 
        plt.show()
        
        if modified:
            fig.savefig(f'results/graphs/{title}', bbox_inches='tight')
        else:
            fig.savefig(f'results/graphs/original/{title}', bbox_inches='tight')
            
    @staticmethod
    def words_distributions(sentences, labels, tokenizer, num_removes = 30):
        c_pos = Counter()
        c_neg = Counter()

        for sentence, label in zip(sentences, labels):
            if label == 1:
                c_pos.update(set(tokenizer.tokenize(sentence)))
            else:
                c_neg.update(set(tokenizer.tokenize(sentence)))

        all_words = list(c_pos.keys())
        all_words.extend(c_neg.keys())
        all_words = set(all_words)

        for word in all_words:
            if word.startswith("##") or c_pos[word]+c_neg[word] < 10:
                del c_pos[word]
                del c_neg[word]
                continue
            c_pos[word] = c_pos[word]/(c_pos[word]+c_neg[word])
            c_neg[word] = c_neg[word]/(c_pos[word]+c_neg[word])

        pos_tokens = list(map(lambda x: x[0], sorted(c_pos.items(), key=lambda item: -item[1])))
        neg_tokens = list(map(lambda x: x[0], sorted(c_neg.items(), key=lambda item: -item[1])))

        return pos_tokens[:num_removes], neg_tokens[:num_removes]

    @staticmethod
    def time_monitor(model_type, ds_name, exps_dict, tokenizer, sentences, labels, top=30, alpha = 0.95):
        """
        compare best topk negative and positive anchors between current result and the
        end result top scores
        """
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        times = pd.read_csv('times.csv', index_col=0)
        pos_percent = sum(labels)/len(labels)
        percents_dict = {}
        for opt, exps in exps_dict.items():
            percents_dict[opt] = ScoreUtils.calculate_time_scores(tokenizer,sentences,exps,labels,[alpha]) 

        for opt in exps_dict.keys():
            final_top_pos = set(percents_dict[opt][alpha]['pos'][100].index[:top])
            final_top_neg = set(percents_dict[opt][alpha]['neg'][100].index[:top])
            pos_results, neg_results = [], []
            percents = percents_dict[opt][alpha]['pos'].keys()
            for i in percents:
                top_pos = set(percents_dict[alpha]['pos'][i].index[:top])
                pos_results.append(len(top_pos.intersection(final_top_pos))/top)
                top_neg = set(percents_dict[alpha]['neg'][i].index[:top])
                neg_results.append(len(top_neg.intersection(final_top_neg))/top)
            time = df.loc[f'{model_type}/{ds_name}/confidence/{opt}'].time
            pos_times = [time*pos_percent*percent/100 for i in percents]
            neg_times = [time*neg_percent*percent/100 for i in percents]
            axs[0].plot(pos_times , pos_results, label = opt)
            axs[1].plot(neg_times , neg_results, label = opt)
        axs[0].set_title('positive')
        axs[1].set_title('negative')
        axs[0].set(xlabel = 'ds percents', ylabel='percents')
        axs[1].set(xlabel = 'ds percents', ylabel='percents')
        axs[0].legend()
        axs[1].legend()