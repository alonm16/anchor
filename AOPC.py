import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
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
    def __init__(self, model, tokenizer, sentences, labels, pos_tokens, neg_tokens, title, num_removes = 30):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
        self.softmax = torch.nn.Softmax()
        self.tokens_method = self._remove_tokens
        self.title = title
        self.num_removes = num_removes
        self.pos_tokens, self.neg_tokens = pos_tokens, neg_tokens
        self.pos_sentences = [tokenizer.tokenize(s) for i, s in enumerate(sentences) if labels[i]==1]
        self.neg_sentences = [tokenizer.tokenize(s) for i, s in enumerate(sentences) if labels[i]==0]
          
    def _remove_tokens(self, idx, tokens, sentences):
        return [['[PAD]' if token in tokens[:idx] else token for token in sentence] for sentence in sentences]
    
    def _replace_tokens(self, idx, tokens, sentences):
        replaced_sentences = []
        for s in sentences:
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
        outputs = self.softmax(self.model(input_ids = input_ids, attention_mask=attention_mask)[0])
        return outputs.detach().cpu().numpy()
    
    def _aopc_predictions(self, sentences_arr, label):
        return np.array([self._predict_scores(sentences)[:, label] for sentences in sentences_arr])
    
    # def _aopc_predictions(self, sentences_arr, label):
    #     predictions = []
    #     for sentences in sentences_arr:
    #         predictions_batch = []
    #         for i in range(0, len(sentences), 50):
    #             predictions_batch.extend(self._predict_scores(sentences[i:i+50])[:, label])
    #         predictions.append(predictions_batch)
    #     return np.array(predictions)
    
    def _aopc_formula(self, orig_predictions, predictions_arr, k):
        N = len(orig_predictions)    
        return sum([sum(orig_predictions - predictions_arr[i])/N for i in range(k+1)])/(k+1)

    def _calc_aopc(self, predictions_arr):
        orig_predictions = predictions_arr[0]
        return [self._aopc_formula(orig_predictions, predictions_arr, k) for k in range(len(predictions_arr))]

    def _aopc_global(self, tokens, sentences, label, removes_arr = None):
        if not removes_arr:
            removes_arr = range(self.num_removes+1)
        removed_sentences_arr = []
        removed_sentences = sentences
        for i in removes_arr:
            removed_sentences = self.tokens_method(i, tokens, removed_sentences)
            removed_sentences_arr.append(removed_sentences) 
        predictions_arr = self._aopc_predictions(removed_sentences_arr, label)
        aopc_scores = self._calc_aopc(predictions_arr)
        return aopc_scores  
        
    def _compare_sorts(self, tokens_method = 'remove', legends = ['normal', 'random', 'reverse', 'baseline'], plotter = AOPC_Plotter.aopc_plot):
        self.tokens_method = self._remove_tokens if tokens_method=='remove' else self._replace_tokens
        pos_scores, neg_scores = [], []
        if 'normal' in legends:
            pos_scores.append(self._aopc_global(self.pos_tokens, self.pos_sentences, 1))
            neg_scores.append(self._aopc_global(self.neg_tokens, self.neg_sentences, 0))
        
        if 'random' in legends:
            random_scores = np.zeros(self.num_removes+1)
            for i in range(5):
                set_seed((i+1)*100)
                shuffled_tokens = self.pos_tokens.copy()
                random.shuffle(shuffled_tokens)
                random_scores += np.array(self._aopc_global(shuffled_tokens, self.pos_sentences, 1))
            random_scores/=5
            pos_scores.append(random_scores)
            
            random_scores = np.zeros(self.num_removes+1)
            for i in range(5):
                set_seed((i+1)*100)
                shuffled_tokens = self.neg_tokens.copy()
                random.shuffle(shuffled_tokens)
                random_scores += np.array(self._aopc_global(shuffled_tokens, self.neg_sentences, 0))
            random_scores/=5
            neg_scores.append(random_scores)
        
        if 'reverse' in legends:
            pos_scores.append(self._aopc_global(self.pos_tokens[::-1], self.pos_sentences, 1))
            neg_scores.append(self._aopc_global(self.neg_tokens[::-1], self.neg_sentences, 0))
            
        if 'baseline' in legends:
            p, n = AOPC.words_distributions(self.pos_sentences+self.neg_sentences, [1]*len(self.pos_sentences)+[0]*len(self.neg_sentences), self.tokenizer)
            pos_scores.append(self._aopc_global(p, self.pos_sentences, 1))
            neg_scores.append(self._aopc_global(n, self.neg_sentences, 0))

        plotter(pos_scores, neg_scores, legends, self.title)
        
    @staticmethod
    def compare_aopcs(model, tokenizer, compare_list, get_scores_fn, sentences, labels, legends, title = "", num_removes = 30, plotter = AOPC_Plotter.aopc_plot):         
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
            aopc = AOPC(model, tokenizer, sentences, labels, pos_tokens_arr[i], neg_tokens_arr[i], title, num_removes = num_removes) 
            pos_scores.append(aopc._aopc_global(aopc.pos_tokens, aopc.pos_sentences, 1))
            neg_scores.append(aopc._aopc_global(aopc.neg_tokens, aopc.neg_sentences, 0))
        
        plotter(pos_scores, neg_scores, legends, title)
        
    @staticmethod
    def compare_all(folder_name, model, tokenizer, sentences, labels, title, num_removes = 30, from_img = [], skip = []):
        def compare_sorts():
            pos_scores, neg_scores = ScoreUtils.get_scores_dict(folder_name, trail_path = "../0.1/scores.xlsx")
            pos_tokens, neg_tokens = list(pos_scores.keys()), list(neg_scores.keys())
            aopc = AOPC(model, tokenizer, sentences, labels, pos_tokens, neg_tokens, title + ' sorts', num_removes)._compare_sorts()
            
        def compare_deltas():
            deltas = [0.1, 0.15, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8]
            get_scores_fn = lambda delta: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../{delta}/scores.xlsx")
            AOPC.compare_aopcs(model, tokenizer, deltas, get_scores_fn, sentences, labels, deltas, f'{title} deltas', num_removes)
        
        def compare_alphas():
            alphas = [0.95, 0.8, 0.65, 0.5]
            get_scores_fn = lambda alpha: ScoreUtils.get_scores_dict(folder_name, trail_path = "../0.1/scores.xlsx", alpha = alpha)
            AOPC.compare_aopcs(model, tokenizer, alphas, get_scores_fn, sentences, labels, alphas, f'{title} alphas', num_removes)
        
        def compare_optimizations():
            optimizations = [str(0.1), 'lossy', 'topk', 'desired']
            get_scores_fn = lambda optimization: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../{optimization}/scores.xlsx")
            AOPC.compare_aopcs(model, tokenizer, optimizations, get_scores_fn, sentences, labels, optimizations, f'{title} optimizations', num_removes)
    
        def compare_aggragations():
            aggragations = ['', '', 'sum_', 'avg_']
            alphas = [0.5, 0.95, None, None]
            legends = ['probabilistic α=0.5', 'probabilistic α=0.95', 'sum', 'avg']
            get_scores_fn = lambda x: ScoreUtils.get_scores_dict(folder_name, folder_name, trail_path = f"../0.1/{x[0]}scores.xlsx", alpha = x[1])
            AOPC.compare_aopcs(model, tokenizer, zip(aggragations, alphas), get_scores_fn, sentences, labels, legends, f'{title} aggragations', num_removes)
        
        def compare_percents():
            percents = [10, 25, 50, 75, 100]
            get_scores_fn = lambda percent: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../0.1/percents/scores-{percent}.xlsx", alpha = 0.95)
            AOPC.compare_aopcs(model, tokenizer, percents, get_scores_fn, sentences, labels, percents, f'{title} percents', num_removes)
        
        def compare_time_percents():
            percents = [10, 25, 50, 75, 100]
            get_scores_fn = lambda percent: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../0.1/percents/scores-{percent}.xlsx", alpha = 0.95)
            AOPC.compare_aopcs(model, tokenizer, percents, get_scores_fn, sentences, labels, percents, f'{title} time-percents', num_removes, plotter=AOPC_Plotter.time_aopc_plot)
            
        compares = {'sorts': compare_sorts, 'deltas': compare_deltas, 'alphas': compare_alphas, 'optimizations': compare_optimizations, 'aggragations': compare_aggragations, 'percents': compare_percents, 'time-percents': compare_time_percents} 

        for c in compares:
            if c in skip:
                continue
            elif c in from_img:
                plt.figure(figsize = (15, 5))
                img = plt.imread(f'results/graphs/{title} {c}.png')
                imgplot = plt.imshow(img)
                plt.axis('off')
                plt.show()
            else:
                compares[c]()
                
    @staticmethod
    def compare_random_aopcs(folder_name, model, tokenizer, seeds, compare_list, sentences, labels, legends, title = "", num_removes = 30): 
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
                aopc = AOPC(model, tokenizer, sentences, labels, pos_tokens_arr[i], neg_tokens_arr[i], title, num_removes = num_removes) 
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
                c_pos.update(sentence)
            else:
                c_neg.update(sentence)

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
    def time_percent_monitor(path, model_type, ds_name, opts, tokenizer, sentences, labels, top=30, alpha = 0.95):
        """
        compare top k anchors during runtime to the final top k of the default running
        """
        get_exps = lambda opt: pickle.load(open(f"{path}/../{opt}/exps_list.pickle", "rb"))
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        times = pd.read_csv('times.csv', index_col=0)
        pos_percent = sum(labels)/len(labels)
        percents_dict = dict()
        percents_dict['0.1'] = ScoreUtils.calculate_time_scores(tokenizer, sentences, get_exps('0.1'), labels,[alpha])
        final_top_pos = set(percents_dict['0.1'][alpha]['pos'][100].index[:top])
        final_top_neg = set(percents_dict['0.1'][alpha]['neg'][100].index[:top])
        
        for opt in opts:
            percents_dict[opt] = ScoreUtils.calculate_time_scores(tokenizer,sentences,get_exps(opt),labels,[alpha]) 

        for opt in opts:
            pos_results, neg_results = [], []
            percents = percents_dict[opt][alpha]['pos'].keys()
            for i in percents:
                top_pos = set(percents_dict[opt][alpha]['pos'][i].index[:top])
                pos_results.append(len(top_pos.intersection(final_top_pos))/top)
                top_neg = set(percents_dict[opt][alpha]['neg'][i].index[:top])
                neg_results.append(len(top_neg.intersection(final_top_neg))/top)
            try:
                time = times.loc[f'{model_type}/{ds_name}/confidence/{opt}'].time
            except:
                time = times.loc[f'{model_type}/{ds_name}/confidence/{opt}-0.1'].time
            pos_times = [time*pos_percent*i/100 for i in percents]
            neg_times = [time*(1-pos_percent)*i/100 for i in percents]
            
            axs[0].plot(pos_times , pos_results, label = opt)
            axs[1].plot(neg_times , neg_results, label = opt)
        axs[0].set_title(f'{model_type} {ds_name} percentage time positive')
        axs[1].set_title(f'{model_type} {ds_name} percentage time negative')
        axs[0].set(xlabel = 'time (minutes)', ylabel='percents')
        axs[1].set(xlabel = 'time (minutes)', ylabel='percents')
        axs[0].legend()
        axs[1].legend()
        
    @staticmethod
    def time_aopc_monitor(path, title, model, model_type, ds_name, opts, tokenizer, sentences, labels, top=30, alpha = 0.95):
        """
        compare best topk negative and positive anchors between current result and the
        end result top scores
        """
        get_exps = lambda opt: pickle.load(open(f"{path}/../{opt}/exps_list.pickle", "rb"))
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        times = pd.read_csv('times.csv', index_col=0)
        pos_percent = sum(labels)/len(labels)
        percents_dict = dict()

        for opt in opts:
            pos_scores, neg_scores = [], []
            percents_dict[opt] = ScoreUtils.calculate_time_scores(tokenizer,sentences,get_exps(opt),labels,[alpha]) 
            percents = percents_dict[opt][alpha]['pos'].keys()
            for i in percents:
                top_pos = list(percents_dict[opt][alpha]['pos'][i].index[:top])
                top_neg = list(percents_dict[opt][alpha]['neg'][i].index[:top])
                aopc =  AOPC(model, tokenizer, sentences, labels, top_pos, top_neg, title, num_removes = top)
                pos_scores.append(2*aopc._aopc_global(aopc.pos_tokens, aopc.pos_sentences, 1, [0, top])[1])
                neg_scores.append(2*aopc._aopc_global(aopc.neg_tokens, aopc.neg_sentences, 0, [0, top])[1])
            
            try:
                time = times.loc[f'{model_type}/{ds_name}/confidence/{opt}'].time
            except:
                time = times.loc[f'{model_type}/{ds_name}/confidence/{opt}-0.1'].time
            pos_times = [time*pos_percent*i/100 for i in percents]
            neg_times = [time*(1-pos_percent)*i/100 for i in percents]
            axs[0].plot(pos_times , pos_scores, label = opt)
            axs[1].plot(neg_times , neg_scores, label = opt)

        axs[0].set_title(f'{model_type} {ds_name} aopc time positive')
        axs[1].set_title(f'{model_type} {ds_name} aopc time negative')
        axs[0].set(xlabel = 'time (minutes)', ylabel='AOPC - global')
        axs[1].set(xlabel = 'time (minutes)', ylabel='AOPC - global')
        axs[0].legend()
        axs[1].legend()