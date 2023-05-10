import torch
from torch.nn.functional import softmax
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
import os
import simple_colors
from score import ScoreUtils
from myUtils import set_seed, get_stopwords
import myUtils
from models.utils import *
colors = [simple_colors.red, simple_colors.blue, simple_colors.cyan, 
          simple_colors.green, simple_colors.magenta, simple_colors.yellow]*3

class AOPC_Plotter:
    @staticmethod
    def aopc_plot(pos_df, neg_df, xlabel, ylabel, hue, legend, title, limiter):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        max_x = max(df[xlabel].max() for df in [pos_df, neg_df])
        max_y = max(df[ylabel].max() for df in [pos_df, neg_df])
        min_y = min(df[ylabel].min() for df in [pos_df, neg_df])
        for ax, df, type_title in zip(axs, [pos_df, neg_df], ['positive', 'negative']): 
            sns.lineplot(data=df, x=xlabel, y=ylabel, hue=hue, legend=legend, ax=ax, palette=sns.color_palette()).set(title=type_title, xlim=(0, max_x), ylim=(min_y, max_y))
            ax.grid()
            if limiter:
                ax.axvline(x = 0.2*max_x, ymin = 0, ymax = max_y, color = 'black', linestyle=':')
        fig.suptitle(title)
        axs[0].legend().remove()
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.show()
        fig.savefig(f'results/graphs/{title}', bbox_inches='tight')
        
    @staticmethod       
    def time_aopc_plot(pos_df, neg_df, xlabel, ylabel, hue, legend, title, limit=False):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        for ax, df, type_title in zip(axs, [pos_df, neg_df], ['positive', 'negative']):
            df = df[df[xlabel].isin([1, 5 ,10, 20 ,30])]
            sns.lineplot(data=df, x=hue, y=ylabel, hue=xlabel, legend=legend, ax = ax, palette=sns.color_palette()).set(title=type_title)
            sns.move_legend(ax, "lower left")
            ax.grid()
        fig.suptitle(title)
        axs[0].legend().remove()
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.show()
        fig.savefig(f'results/graphs/{title}', bbox_inches='tight')

class AOPC:
    def __init__(self, path, tokenizer, delta=0.1, alpha=0.5, num_removes = 30, base_opt=None):
        self.opt_prefix = f'{base_opt}-' if base_opt else ''
        self.model_type, self.ds_name = path.split('/')[-4:-2]
        self.sentences = pickle.load(open(f"{path}42/{self.opt_prefix}{delta}/anchor_examples.pickle", "rb" ))
        self.labels = pickle.load(open(f"{path}42/{self.opt_prefix}0.1/labels.pickle", "rb" ))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(f'models/{self.model_type}/{self.ds_name}/model').to(self.device).eval()
        myUtils.model = self.model
        self.title = f"{self.ds_name} dataset"
        self.path = path
        self.tokenizer = tokenizer
        self.tokens_method = self._remove_tokens
        self.delta = str(delta)
        self.alpha = alpha
        self.num_removes = num_removes
        self.pos_tokens, self.neg_tokens = None, None
        self.pos_sentences = [tokenizer.tokenize(s) for i, s in enumerate(self.sentences) if self.labels[i]==1]
        self.neg_sentences = [tokenizer.tokenize(s) for i, s in enumerate(self.sentences) if self.labels[i]==0]
        self.opts = [str(delta), f'stop-words-{delta}', f'topk-{delta}', f'desired-{delta}', f'masking-{delta}', f'stop-words-masking-{delta}', 'stop-words-0.5', 'stop-words-topk-0.5', f'stop-words-topk-masking-0.5', f'stop-words-topk-desired-masking-0.5']
        #self.opts = [str(delta),  '0.5', f'topk-{delta}', 'topk-0.5', f'desired-{delta}', f'masking-{delta}', f'topk-masking-{delta}', f'topk-masking-0.5', f'topk-desired-masking-{delta}', f'topk-desired-masking-0.5']
        
    def set_tokens(self, pos_tokens, neg_tokens):
        self.pos_tokens, self.neg_tokens = pos_tokens, neg_tokens
        
    def _remove_tokens(self, idx, tokens, sentences):
        return [['[PAD]' if token in tokens[:idx] else token for token in sentence] for sentence in sentences]
    
    def _replace_tokens(self, idx, tokens, sentences):
        unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
        replaced_sentences = []
        for s in sentences:
            tokenized_sentence = copy.deepcopy(s)
            for i in range(len(tokenized_sentence)):
                token = tokenized_sentence[i]
                if token in tokens[:idx]:
                    tokenized_sentence[i] = '[MASK]'
                    sentence = self.tokenizer.decode(self.tokenizer.encode(tokenized_sentence))
                    results = unmasker(sentence, top_k=2)
                    for r in results:
                        if r['token_str']!=token:
                            tokenized_sentence[i] = r['token_str']
                            break

            replaced_sentences.append(tokenized_sentence)
        return replaced_sentences      

    @torch.no_grad()
    def _predict_scores(self, sentences):
        pad = max(len(s) for s in sentences)
        input_ids = [[101] +[self.tokenizer.vocab[token] for token in tokens] + [102] + [0]*(pad-len(tokens)) for tokens in sentences]
        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = [[1]*(len(tokens)+2)+[0]*(pad-len(tokens)) for tokens in sentences]
        attention_mask = torch.tensor(attention_mask, device=self.device)
        outputs = softmax(self.model(input_ids = input_ids, attention_mask=attention_mask)[0])
        #outputs = softmax(self.model(input_ids)[0])
        return outputs.cpu().numpy()
    
    def _aopc_predictions(self, sentences_arr, label):
        return np.array([self._predict_scores(sentences)[:, label] for sentences in sentences_arr])
       
    @staticmethod
    def _aopc_formula(orig_predictions, predictions_arr, k):
        N = len(orig_predictions)    
        return sum([sum(orig_predictions - predictions_arr[i])/N for i in range(k+1)])/(k+1)

    @staticmethod
    def _calc_aopc(predictions_arr):
        orig_predictions = predictions_arr[0]
        return [AOPC._aopc_formula(orig_predictions, predictions_arr, k) for k in range(len(predictions_arr))]

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
        
    @staticmethod
    def words_distributions(sentences, labels, tokenizer, num_removes = 30):
        stopwords = get_stopwords()
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
            if word.startswith("##") or c_pos[word]+c_neg[word] < 10 or word in stopwords:
                del c_pos[word]
                del c_neg[word]
                continue
            c_pos[word] = c_pos[word]/(c_pos[word]+c_neg[word])
            c_neg[word] = c_neg[word]/(c_pos[word]+c_neg[word])

        pos_tokens = list(map(lambda x: x[0], sorted(c_pos.items(), key=lambda item: -item[1])))
        neg_tokens = list(map(lambda x: x[0], sorted(c_neg.items(), key=lambda item: -item[1])))

        return pos_tokens[:num_removes], neg_tokens[:num_removes]
    
    def _compare_sorts(self, tokens_method = 'remove', legends = ['normal', 'random', 'reverse', 'baseline'], hue = 'sorts', xlabel="# of features removed", ylabel="AOPC-global", plotter = AOPC_Plotter.aopc_plot):
        self.tokens_method = self._remove_tokens if tokens_method=='remove' else self._replace_tokens
        pos_df = pd.DataFrame(columns = [xlabel, ylabel, hue])
        neg_df = pd.DataFrame(columns = pos_df.columns)
        pos_tokens_arr, neg_tokens_arr = [], []
        num_removes = self.num_removes
        
        if 'normal' in legends:
            pos_scores, neg_scores = self._aopc_global(self.pos_tokens, self.pos_sentences, 1), self._aopc_global(self.neg_tokens, self.neg_sentences, 0)
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat('normal', num_removes+1))), columns=pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat('normal', num_removes+1))), columns=neg_df.columns)])
            pos_tokens_arr.append(self.pos_tokens)
            neg_tokens_arr.append(self.neg_tokens)

        if 'random' in legends:
            random.seed(self.seed)
            shuffled_pos, shuffled_neg = self.pos_tokens.copy(), self.neg_tokens.copy()
            random.shuffle(shuffled_pos), random.shuffle(shuffled_neg)
            pos_scores, neg_scores = self._aopc_global(shuffled_pos, self.pos_sentences, 1), self._aopc_global(shuffled_neg, self.neg_sentences, 0)
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat('random', num_removes+1))), columns=pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat('random', num_removes+1))), columns=pos_df.columns)])
            pos_tokens_arr.append(shuffled_pos)
            neg_tokens_arr.append(shuffled_neg)

        if 'reverse' in legends:
            pos_scores, neg_scores = self._aopc_global(self.pos_tokens[::-1], self.pos_sentences, 1), self._aopc_global(self.neg_tokens[::-1], self.neg_sentences, 0)
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat('reverse', num_removes+1))), columns=pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat('reverse', num_removes+1))), columns=neg_df.columns)])
            pos_tokens_arr.append(self.pos_tokens[::-1])
            neg_tokens_arr.append(self.neg_tokens[::-1])
            
        if 'baseline' in legends:
            p, n = self.words_distributions(self.pos_sentences+self.neg_sentences, [1]*len(self.pos_sentences)+[0]*len(self.neg_sentences), self.tokenizer)
            pos_scores, neg_scores = self._aopc_global(p, self.pos_sentences, 1), self._aopc_global(n, self.neg_sentences, 0)
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat('baseline', num_removes+1))), columns=pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat('baseline', num_removes+1))), columns=neg_df.columns)])
            pos_tokens_arr.append(p)
            neg_tokens_arr.append(n)
        
        if self.seed==42:
            self.print_tokens(legends, pos_tokens_arr, neg_tokens_arr)
            os.makedirs(f"{self.seed_path}tokens", exist_ok=True)
            for l in legends:
                pickle.dump(pos_tokens_arr[legends.index(l)], open(f"{self.seed_path}tokens/{l}_pos_tokens.pickle", "wb"))
                pickle.dump(neg_tokens_arr[legends.index(l)], open(f"{self.seed_path}tokens/{l}_neg_tokens.pickle", "wb"))
                
        return pos_df, neg_df, xlabel, ylabel, hue, legends, self.title + f' {hue}', plotter, 'normal'
        
    def compare_aopcs(self, compare_list, get_scores_fn, legends, hue, xlabel="# of features removed", ylabel="AOPC-global", plotter=AOPC_Plotter.aopc_plot, normalizer = None): 
        num_removes = self.num_removes
        pos_tokens_arr = []
        neg_tokens_arr = []
        for item in compare_list:
            pos_scores, neg_scores = get_scores_fn(item)
            pos_tokens, neg_tokens = list(pos_scores.keys()), list(neg_scores.keys())
            pos_tokens_arr.append(pos_tokens)
            neg_tokens_arr.append(neg_tokens)
        
        pos_df = pd.DataFrame(columns = [xlabel, ylabel, hue])
        neg_df = pd.DataFrame(columns = pos_df.columns)
        for i in range(len(legends)):
            self.set_tokens(pos_tokens_arr[i], neg_tokens_arr[i])
            pos_scores, neg_scores = self._aopc_global(self.pos_tokens, self.pos_sentences, 1), self._aopc_global(self.neg_tokens, self.neg_sentences, 0)
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat(str(legends[i]), num_removes+1))), columns=pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat(str(legends[i]), num_removes+1))), columns=neg_df.columns)])
        
        if self.seed==42:
            self.print_tokens(legends, pos_tokens_arr, neg_tokens_arr)
            os.makedirs(f"{self.seed_path}tokens", exist_ok=True)
            for l in legends:
                pickle.dump(pos_tokens_arr[legends.index(l)], open(f"{self.seed_path}tokens/{l}_pos_tokens.pickle", "wb"))
                pickle.dump(neg_tokens_arr[legends.index(l)], open(f"{self.seed_path}tokens/{l}_neg_tokens.pickle", "wb"))
            
        return pos_df, neg_df, xlabel, ylabel, hue, legends, self.title + f' {hue}', plotter, normalizer
        
    def compare_sorts(self, **kwargs):
        pos_scores, neg_scores = ScoreUtils.get_scores_dict(self.seed_path, trail_path = f"{self.delta}/scores.xlsx", alpha = self.alpha)
        pos_tokens, neg_tokens = list(pos_scores.keys()), list(neg_scores.keys())
        self.set_tokens(pos_tokens, neg_tokens)
        return self._compare_sorts()
    
    def compare_deltas(self, **kwargs):
        deltas = [0.1, 0.15, 0.2, 0.35, 0.5]#, 0.6, 0.7, 0.8]
        get_scores_fn = lambda cur_delta: ScoreUtils.get_scores_dict(self.seed_path, trail_path = f"{cur_delta}/scores.xlsx", alpha = self.alpha)
        return self.compare_aopcs(deltas, get_scores_fn, deltas, 'delta', normalizer=self.delta)
    
    def compare_alphas(self, **kwargs):
        alphas = [0.95, 0.8, 0.65, 0.5]
        get_scores_fn = lambda alpha: ScoreUtils.get_scores_dict(self.seed_path, trail_path = f"{self.delta}/scores.xlsx", alpha = alpha)
        return self.compare_aopcs(alphas, get_scores_fn, alphas, 'alpha', normalizer=0.95)
    
    def compare_optimizations(self, **kwargs):
        optimizations = kwargs['opts'] if 'opts' in kwargs else self.opts
        get_scores_fn = lambda optimization: ScoreUtils.get_scores_dict(self.seed_path, trail_path = f"{optimization}/scores.xlsx", alpha = self.alpha)
        return self.compare_aopcs(optimizations, get_scores_fn, optimizations, 'optimization', normalizer=self.delta)
    
    def compare_aggregations(self, **kwargs):
        if 'agg_params' not in kwargs:
            aggregations = ['', '', 'sum_', 'avg_', f'avg_{2e-3}_', f'avg_{3e-3}_', f'avg_{4e-3}_']
            alphas = [0.5, 0.95, None, None, None, None, None]
            legends = ['probabilistic α=0.5', 'probabilistic α=0.95', 'sum', 'avg', 'avg_2e-3', 'avg_3e-3', 'avg_4e-3']
        else:
            aggregations, alphas, legends = kwargs['agg_params']
        get_scores_fn = lambda x: ScoreUtils.get_scores_dict(self.seed_path, trail_path=f"{self.delta}/{x[0]}scores.xlsx", alpha = x[1])
        return self.compare_aopcs(zip(aggregations, alphas), get_scores_fn, legends, 'aggregation', normalizer='probabilistic α=0.95')
        
    def compare_percents(self, **kwargs):
        percents = [5, 10, 25, 50, 75, 100]
        get_scores_fn = lambda percent: ScoreUtils.get_scores_dict(self.seed_path, trail_path = f"{self.delta}/percents/scores-{percent}.xlsx", alpha = self.alpha)
        return self.compare_aopcs(percents, get_scores_fn, percents, 'percent', normalizer=100)

    def compare_percents_remove(self, **kwargs):
        percents = [5, 10, 25, 50, 75, 100]
        get_scores_fn = lambda percent: ScoreUtils.get_scores_dict(self.seed_path, trail_path = f"{self.delta}/percents/scores-{percent}.xlsx", alpha = self.alpha)
        return self.compare_aopcs(percents, get_scores_fn, percents, 'percents-remove', plotter=AOPC_Plotter.time_aopc_plot, normalizer=100)
          
    def time_percent(self, **kwargs):
        opts = kwargs['opts'] if 'opts' in kwargs else self.opts
        return self.time_percent_monitor(opts, alpha = self.alpha)
        
    def time_aopc(self, **kwargs):
        opts = kwargs['opts'] if 'opts' in kwargs else self.opts
        return self.time_aopc_monitor(opts, alpha = self.alpha)
    
    def compare_all(self, seeds=[42, 84, 126, 168, 210], from_img=[], skip=[], only=None, verbose= True, **kwargs): 
        if not verbose:
            global print
            print = lambda *x: None
        compares = {'sorts': self.compare_sorts, 'delta': self.compare_deltas, 'alpha': self.compare_alphas, 'optimization': self.compare_optimizations, 'aggregation': self.compare_aggregations, 'percent': self.compare_percents, 'percents-remove': self.compare_percents_remove, 'percents time': self.time_percent, 'aopc time': self.time_aopc}
        
        normalizers =  dict(zip(compares.keys(),['normal', 0.1, 0.95, self.delta, 'probabilistic α=0.95', 100, 100, self.delta, self.delta]))
        
        for c in compares:
            if only and c not in only:
                continue
            if c in skip:
                continue
            elif c in from_img:
                pos_df = pd.read_csv(f'{self.path}/{self.opt_prefix}{c}_pos_aopc.csv', index_col=0)
                neg_df = pd.read_csv(f'{self.path}/{self.opt_prefix}{c}_neg_aopc.csv', index_col=0)
                legends = list(pos_df.iloc[:, -1].unique())
                xlabel, ylabel, hue = pos_df.columns
                normalizer = normalizers[c]
                jump = len(pos_df)//len(seeds)//len(legends)
                jumps = list(range(jump-1, len(pos_df)//len(legends), jump))
                pos_normalizer = pos_df[pos_df[hue].astype('str')==str(normalizer)].iloc[jumps, 1].mean()
                jump = len(neg_df)//len(seeds)//len(legends)
                jumps = list(range(jump-1, len(neg_df)//len(legends), jump))
                neg_normalizer = neg_df[neg_df[hue].astype('str')==str(normalizer)].iloc[jumps, 1].mean()
                pos_df.iloc[:, 1]/=pos_normalizer
                neg_df.iloc[:, 1]/=neg_normalizer
                c_title = self.title + f' {hue}'
                plotter = AOPC_Plotter.aopc_plot if c!='percents-remove' else AOPC_Plotter.time_aopc_plot
                pos_tok_arr = [pickle.load(open(f"{self.path}42/{self.opt_prefix}tokens/{l}_pos_tokens.pickle", "rb")) for l in legends]
                neg_tok_arr = [pickle.load(open(f"{self.path}42/{self.opt_prefix}tokens/{l}_neg_tokens.pickle", "rb")) for l in legends]
                self.print_tokens(legends, pos_tok_arr, neg_tok_arr)
                limiter=False
                if c in ['percents time', 'aopc time']:
                    limiter=True
                    c_title = self.title + f' {c.split()[0]} evaluation'
                plotter(pos_df, neg_df, xlabel, ylabel, hue, legends, c_title, limiter)
                
            else:
                pos_df, neg_df = pd.DataFrame(), pd.DataFrame()
                for seed in seeds:
                    self.seed = seed
                    self.seed_path = self.path+f'{seed}/{self.opt_prefix}'
                    cur_pos, cur_neg, xlabel, ylabel, hue, legends, c_title, plotter, normalizer = compares[c](**kwargs)
                    pos_df, neg_df = pd.concat([pos_df, cur_pos]), pd.concat([neg_df, cur_neg])
                pos_df.to_csv(f'{self.path}/{self.opt_prefix}{c}_pos_aopc.csv')
                neg_df.to_csv(f'{self.path}/{self.opt_prefix}{c}_neg_aopc.csv')
                
                if normalizer:
                    jump = len(pos_df)//len(seeds)//len(legends)
                    jumps = list(range(jump-1, len(pos_df)//len(legends), jump))
                    pos_normalizer = pos_df[pos_df[hue].astype('str') ==str(normalizer)].iloc[jumps, 1].mean()
                    jump = len(neg_df)//len(seeds)//len(legends)
                    jumps = list(range(jump-1, len(neg_df)//len(legends), jump))
                    neg_normalizer = neg_df[neg_df[hue].astype('str') ==str(normalizer)].iloc[jumps, 1].mean()
                    pos_df.iloc[:, 1]/=pos_normalizer
                    neg_df.iloc[:, 1]/=neg_normalizer
                limiter=False
                if c in ['percents time', 'aopc time']:
                    limiter=True
                plotter(pos_df, neg_df, xlabel, ylabel, hue, legends, c_title, limiter)

                
    def time_percent_monitor(self, opts, alpha=0.95):
        """
        compare top k anchors during runtime to the final top k of the default running
        """
        top = self.num_removes
        get_exps = lambda opt: pickle.load(open(f"{self.seed_path}{opt}/exps_list.pickle", "rb"))
        times = pd.read_csv('times.csv', index_col=0)
        pos_percent = sum(self.labels)/len(self.labels)
        default_res = ScoreUtils.calculate_time_scores(self.tokenizer, self.sentences, get_exps(self.delta), self.labels, [alpha])[alpha]
        final_top_pos = set(default_res['pos'][100].index[:top])
        final_top_neg = set(default_res['neg'][100].index[:top])
        pos_df = pd.DataFrame(columns = ['time (minutes)', "percents", "optimization"])
        neg_df = pd.DataFrame(columns = pos_df.columns)

        for opt in opts:
            percents_dict = ScoreUtils.calculate_time_scores(self.tokenizer, self.sentences, get_exps(opt), self.labels, [alpha])[alpha]
            pos_results, neg_results = [], []
            percents = percents_dict['pos'].keys()
            for i in percents:
                top_pos = set(percents_dict['pos'][i].index[:top])
                pos_results.append(len(top_pos.intersection(final_top_pos))/top)
                top_neg = set(percents_dict['neg'][i].index[:top])
                neg_results.append(len(top_neg.intersection(final_top_neg))/top)
            
            #only time of one seed so the aggregation of lineplot will work
            time = times.loc[f'mp/{self.model_type}/{self.ds_name}/confidence/42/{self.opt_prefix}{opt}'].time
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip([time*pos_percent*i/100 for i in percents], pos_results, np.repeat(opt, len(pos_results)))), columns = pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip([time*(1-pos_percent)*i/100 for i in percents], neg_results, np.repeat(opt, len(neg_results)))), columns = neg_df.columns)])
            
        return pos_df, neg_df, 'time (minutes)', 'percents', "optimization", opts, f'{self.ds_name} dataset percents evaluation', AOPC_Plotter.aopc_plot, self.delta
                
    def time_aopc_monitor(self, opts, alpha=0.95):
        """
        compare best topk negative and positive anchors between current result and the
        end result top scores
        """
        top = self.num_removes
        get_exps = lambda opt: pickle.load(open(f"{self.seed_path}{opt}/exps_list.pickle", "rb"))
        times = pd.read_csv('times.csv', index_col=0)
        pos_percent = sum(self.labels)/len(self.labels)
        pos_df = pd.DataFrame(columns = ['time (minutes)', "AOPC-global", "optimization"])
        neg_df = pd.DataFrame(columns = pos_df.columns)
        pos_tok_arr, neg_tok_arr = [], []

        for opt in opts:
            pos_results, neg_results = [], []
            percents_dict = ScoreUtils.calculate_time_scores(self.tokenizer, self.sentences, get_exps(opt), self.labels, [alpha])[alpha]
            percents = percents_dict['pos'].keys()
            for i in percents:
                top_pos = list(percents_dict['pos'][i].index[:top])
                top_neg = list(percents_dict['neg'][i].index[:top])
                self.set_tokens(top_pos, top_neg)
                pos_results.append(2*self._aopc_global(top_pos, self.pos_sentences, 1, [0, top])[1])
                neg_results.append(2*self._aopc_global(top_neg, self.neg_sentences, 0, [0, top])[1])
            pos_tok_arr.append(top_pos)
            neg_tok_arr.append(top_neg)
                
            time = times.loc[f'mp/{self.model_type}/{self.ds_name}/confidence/42/{self.opt_prefix}{opt}'].time
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip([time*pos_percent*i/100 for i in percents], pos_results, np.repeat(opt, len(pos_results)))), columns = pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip([time*(1-pos_percent)*i/100 for i in percents], neg_results, np.repeat(opt, len(neg_results)))), columns = neg_df.columns)])

        if self.seed==42:
            self.print_tokens(opts, pos_tok_arr, neg_tok_arr)
                
        return pos_df, neg_df, 'time (minutes)', 'AOPC-global', "optimization", opts, f'{self.ds_name} dataset aopc evaluation', AOPC_Plotter.aopc_plot, self.delta
    
    @staticmethod
    def print_tokens(legends, pos_tokens_arr, neg_tokens_arr):
        print('pos')
        for l in legends:
            print(colors[legends.index(l)](f'{l}:'), f'{pos_tokens_arr[legends.index(l)][:10]}')
        print('\nneg')
        for l in legends:
            print(colors[legends.index(l)](f'{l}:'), f'{neg_tokens_arr[legends.index(l)][:10]}')