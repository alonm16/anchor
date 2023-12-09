import torch
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import Counter, defaultdict
from functools import reduce
from transformers import pipeline
import copy
import pandas as pd
import seaborn as sns
import os
from simple_colors import *
from score import ScoreUtils
from myUtils import set_seed, get_stopwords
import myUtils
from models.utils import *
colors = [red, blue, cyan, green, magenta, yellow]*3

class AOPC_Plotter:
    @staticmethod
    def aopc_plot(pos_df, neg_df, xlabel, ylabel, hue, legend, title, limiter):
        fig, axs = plt.subplots(1, 2, figsize=(5, 1.805))
        max_x = max(df[xlabel].max() for df in [pos_df, neg_df])
        max_y = max(df[ylabel].max() for df in [pos_df, neg_df])
        min_y = min(df[ylabel].min() for df in [pos_df, neg_df])
        for ax, df, type_title in zip(axs, [pos_df, neg_df], ['positive', 'negative']): 
            sns.lineplot(data=df, x=xlabel, y=ylabel, hue=hue, legend=legend, ax=ax, palette=sns.color_palette('colorblind')).set(title=type_title, xlim=(0, max_x), ylim=(min_y, max_y))
            ax.grid()
            ax.set(xlabel=None)
            if limiter:
                ax.axvline(x = 0.2*max_x, ymin = 0, ymax = max_y, color = 'black', linestyle=':')
            if 'time aggregation' in title:
                ax.axhline(y=1, color = 'black', linestyle=':')
        fig.suptitle(title, y=1.08)
        fig.supxlabel(xlabel, y=-0.092, fontsize = 'medium')
        axs[1].legend().remove()
        axs[1].axes.get_yaxis().get_label().set_visible(False)
        axs[1].set(yticklabels=[])
        ncols = 6 if len(legend)<=6 else 4
        axs[0].legend(title = hue, loc='upper center', bbox_to_anchor=(1., -0.22),
            fancybox=True, shadow=True, ncol=ncols, fontsize='10')
        plt.show()
        if 'α' in title:
            title = title.replace('α', 'alpha')
        if 'δ' in title:
            title = title.replace('δ', 'delta')
        fig.savefig(f'results/graphs/{title}', bbox_inches='tight')
        
    @staticmethod
    def aopc_plot_time(pos_df, neg_df, xlabel, ylabel, hue, legend, title, limiter):
        modify = lambda x: x.replace('topk-desired-masking', 'all')
        pos_df[hue] = pos_df[hue].apply(modify)
        neg_df[hue] = neg_df[hue].apply(modify)
        change = lambda df, delta: df[df['optimization'].str.contains(delta)]
        df_1_pos, df_5_pos = change(pos_df, '0.1'), change(pos_df, '0.5')
        df_1_neg, df_5_neg = change(neg_df, '0.1'), change(neg_df, '0.5')
        shorten = lambda x: 'default' if len(x) < 4 else x[:-4]
        dfs = [df_1_pos, df_5_pos, df_1_neg, df_5_neg]
        for df in dfs:
            df['optimization'] = df['optimization'].apply(shorten)
        fig, axs = plt.subplots(1, 4, figsize=(11, 3))
        max_x = max(df[xlabel].max() for df in dfs)
        max_y = max(df[ylabel].max() for df in dfs)
        min_y = min(df[ylabel].min() for df in dfs)
        max_x_5 = max(df[xlabel].max() for df in [df_5_pos, df_5_neg])
        
        for i, df, type_title in zip(range(len(axs)), dfs, ['δ=0.1 positive', 'δ=0.5 positive', 'δ=0.1 negative', 'δ=0.5 negative']): 
            lim = max_x if '1' in type_title else max_x_5
            sns.lineplot(data=df, x=xlabel, y=ylabel, hue=hue, legend=legend, ax=axs[i], palette=sns.color_palette('colorblind')).set(xlim=(0, lim), ylim=(min_y, max_y))
            axs[i].grid()
            axs[i].set(xlabel=None)
            axs[i].set_title(type_title, fontsize = 14)

            if i>0:
                axs[i].legend().remove()
                axs[i].axes.get_yaxis().get_label().set_visible(False)
                axs[i].set(yticklabels=[])
            if limiter:
                axs[i].axvline(x = 0.2*max_x, ymin = 0, ymax = max_y, color='black', linestyle=':')
            if 'time aggregation' in title:
                axs[i].axhline(y=1, color = 'black', linestyle=':')
              
        fig.suptitle(title, y=1.07, fontsize = '18')
        fig.supxlabel(xlabel, y=-0.05, fontsize = '16')
        ncols = 8
        axs[0].set_ylabel(ylabel.replace("-global",""), fontsize = '16')
        axs[0].legend(title = hue, loc='upper center', bbox_to_anchor=(2.3, -0.19),
            fancybox=True, shadow=True, ncol=ncols, fontsize='16')
        plt.setp(axs[0].get_legend().get_title(), fontsize='15')
        plt.show()
        fig.savefig(f'results/graphs/{title}', bbox_inches='tight')
        
    @staticmethod
    def aopc_plot_helper(dfs, xlabel, ylabel, hue, legend, title, datasets):
        dark = sns.color_palette('dark')
        colorblind = sns.color_palette('colorblind')
        colors = [colorblind[9], colorblind[2], dark[7], dark[4], dark[1], colorblind[4], colorblind[1]]
        ax_titles = ['+', '-']*(len(dfs)//2)
        fig, axs = plt.subplots(1, len(dfs), figsize=(18, 2.3))
        max_xs = [df[xlabel].max() for df in dfs]
        max_y = max(df[ylabel].max() for df in dfs)
        min_y = min(0, min(df[ylabel].min() for df in dfs))#min(df[ylabel].min() for df in dfs)
        
        real_x = xlabel.replace("# of features removed", 'k').replace('time (minutes)', 'time (min)')
        
        for i, df, type_title in zip(range(len(axs)), dfs, ax_titles): 
            sns.lineplot(data=df, x=xlabel, y=ylabel, hue=hue, legend=legend, ax=axs[i], palette=sns.color_palette(colors)).set(xlim=(0, max_xs[i]), ylim=(min_y, max_y))
            axs[i].grid()
            axs[i].set_xlabel(real_x, fontsize = 12)
            axs[i].set_title(datasets[i//2] + ' ' + type_title, fontsize = 14)

            if i>0:
                axs[i].legend().remove()
                axs[i].axes.get_yaxis().get_label().set_visible(False)
                axs[i].set(yticklabels=[])
            
            if title!='Aggregations':
                axs[i].axvline(x = 0.2*max_xs[i], ymin = 0, ymax = max_y, color='black', linestyle=':')
            
            if 'time aggregation' in title:
                axs[i].axhline(y=1, color = 'black', linestyle=':')
              
        #fig.suptitle(title, y=1.1, fontsize = '18'),
        #fig.supxlabel(xlabel, y=-0.1, fontsize = '16')
       
        ncols = 8
        axs[0].set_ylabel(ylabel.replace("AOPC-global","$AOPC^k$").replace('percents', 'ratio'), fontsize = '16')
        axs[0].legend(title = hue, loc='upper center', bbox_to_anchor=(4.5, -0.22),
            fancybox=True, shadow=True, ncol=ncols, fontsize='16')
        plt.setp(axs[0].get_legend().get_title(), fontsize='0')
        plt.show()
        fig.savefig(f'results/graphs/{title}', bbox_inches='tight')
        
    @staticmethod
    def aopc_plot_opt(pos_dfs, neg_dfs, xlabel, ylabel, hue, legend, title, datasets, values = ['0.1', '0.5', 'topk-desired-masking-0.1', 'topk-desired-masking-0.5']):
        modify = lambda x: x.replace('topk-desired-masking-', 'Optimized ').replace('0.1', '$\delta=0.1$').replace('0.5', '$\delta=0.5$')
        modify2 = lambda x: x.replace('$\delta=0.1$', '').replace('()','').replace('-','').replace('masking', 'Masking').replace('topk', 'Filtering').replace('desired', 'Confidence')
        modify3 = lambda x: x if len(x) > 1 else 'Default' 
        dfs = []
        for i in range(len(pos_dfs)):
            dfs.extend([pos_dfs[i], neg_dfs[i]])
        for i in range(len(dfs)):
            dfs[i] = dfs[i][dfs[i][hue].isin(values)]
            dfs[i][hue] = dfs[i][hue].apply(modify)
            if all('0.1' in x for x in dfs[i][hue]):
                 dfs[i][hue] =  dfs[i][hue].apply(modify2).apply(modify3)
        
        AOPC_Plotter.aopc_plot_helper(dfs, xlabel, ylabel, hue, legend, title, datasets)
        
    @staticmethod
    def aopc_plot_agg(pos_dfs, neg_dfs, xlabel, ylabel, hue, legend, title, datasets):
        dfs = []
        for i in range(len(pos_dfs)):
            dfs.extend([pos_dfs[i], neg_dfs[i]])
            
        modify = lambda x: x.replace('α=0.5', '')
        for i in range(len(dfs)):
            dfs[i] = dfs[i][dfs[i][hue]!='$\mathcal{G}_{\mathsf{pr}}$ α=0.95']
            dfs[i][hue] = dfs[i][hue].apply(modify)
            
        AOPC_Plotter.aopc_plot_helper(dfs, xlabel, ylabel, hue, legend, title, datasets)
        
    @staticmethod
    def plot_single(dfs, xlabel, ylabel, hue, legend, title, datasets):
        modify = lambda x: x.replace('topk-desired-masking-', 'Optimized ').replace('0.1', '$\delta=0.1$').replace('0.5', '$\delta=0.5$')
        modify2 = lambda x: x.replace('$\delta=0.1$', '').replace('()','').replace('-','').replace('masking', 'Masking').replace('topk', 'Filtering').replace('desired', 'Confidence')
        values = ['$\mathcal{G}_{\mathsf{sq}}$', 'topk-desired-masking-0.5', '$\mathcal{G}_{\mathsf{av}}$']
        for i in range(len(dfs)):
            dfs[i] = dfs[i][dfs[i][hue].isin(values)]
            dfs[i][hue] = dfs[i][hue].apply(modify)
            if all('0.1' in x for x in dfs[i][hue]):
                 dfs[i][hue] =  dfs[i][hue].apply(modify2)

        dark = sns.color_palette('dark')
        colorblind = sns.color_palette('colorblind')
        colors = [dark[7], dark[9], dark[3], colorblind[2], dark[4], dark[1], colorblind[4], colorblind[1]]
        ax_titles = ['-']*len(dfs)
        fig, ax = plt.subplots(1, len(dfs), figsize=(2, 1.5))
        max_xs = [df[xlabel].max() for df in dfs]
        max_y = max(df[ylabel].max() for df in dfs)
        min_y = 0#min(df[ylabel].min() for df in dfs)
        
        real_x = xlabel.replace("# of features removed", 'k').replace('time (minutes)', 'time (min)')
        
        for i, df, type_title in zip([0], dfs, ax_titles): 
            df[xlabel] = 60*df[xlabel]
            sns.lineplot(data=df, x=xlabel, y=ylabel, hue=hue, legend=legend, ax=ax, palette=sns.color_palette(colors)).set(xlim=(0, max_xs[i]), ylim=(min_y, max_y))
            ax.grid()
            ax.set_xlabel(real_x, fontsize = 12)

            ax.legend().remove()

            #ax.axvline(x = 0.2*max_xs[i], ymin = 0, ymax = max_y, color='black', linestyle=':')
            
        ax.set_ylabel(ylabel.replace("AOPC-global","AOPC$^k$").replace('percents', 'ratio'), fontsize = '12')
        #ax.set_xlabel('time(sec)')
        ax.set_xlabel('')
        #ax.set_ylim(0, 1.2)
        ax.set_xlim(0, 60)
        #ax.legend(title ='', loc='upper right', bbox_to_anchor=(1.6, 1.05),
        #    fancybox=True, shadow=True, ncol=1, fontsize='10')
        plt.yticks(np.arange(0, 1.6, 0.2))
        plt.show()
        fig.savefig(f'results/graphs/intro', bbox_inches='tight')
        
class AOPC:
    def __init__(self, path, tokenizer, delta=0.1, alpha=0.5, num_removes = 20, base_opt=None):
        self.opt_prefix = f'{base_opt}-' if base_opt else ''
        self.model_type, self.ds_name = path.split('/')[-4:-2]
        self.sentences = pickle.load(open(f"{path}42/{self.opt_prefix}{delta}/anchor_examples.pickle", "rb" ))
        self.labels = pickle.load(open(f"{path}42/{self.opt_prefix}0.1/labels.pickle", "rb"))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(f'models/{self.model_type}/{self.ds_name}/model').to(self.device).eval()
        myUtils.model = self.model
        self.title = f"{self.ds_name} dataset"
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=False)
        self.tokens_method = self._remove_tokens
        self.delta = str(delta)
        self.alpha = alpha
        self.num_removes = num_removes
        self.pos_tokens, self.neg_tokens = None, None
        self.pos_sentences = [s for i, s in enumerate(self.sentences) if self.labels[i]==1]
        self.neg_sentences = [s for i, s in enumerate(self.sentences) if self.labels[i]==0]
        self.opts = [str(delta), f'stop-words-{delta}', f'topk-{delta}', f'desired-{delta}', f'masking-{delta}', f'stop-words-masking-{delta}', 'stop-words-0.5', 'stop-words-topk-0.5', f'stop-words-topk-masking-0.5', f'stop-words-topk-desired-masking-0.5']
        if self.opt_prefix != '':
            self.opts = [str(delta),  '0.5', f'topk-{delta}', 'topk-0.5', f'desired-{delta}', 'desired-0.5', f'masking-{delta}', 'masking-0.5', f'topk-desired-masking-{delta}', 'topk-desired-masking-0.5']
        self.model_tokenizer = tokenizer
        self.predictions = pickle.load(open(f"{path}42/{self.opt_prefix}0.1/predictions.pickle", "rb" ))
        
    def set_tokens(self, pos_tokens, neg_tokens):
        self.pos_tokens, self.neg_tokens = pos_tokens, neg_tokens
        
    def _remove_tokens(self, idx, tokens, sentences):
        removed_sentences = [self.tokenizer.tokenize(s) for s in sentences]
        removed_sentences =  [['[PAD]' if token in tokens[:idx] else token for token in sentence] for sentence in removed_sentences]
        ids = [[self.tokenizer.vocab[token] for token in tokens] 
                           for tokens in removed_sentences]
        return [self.tokenizer.decode(s) for s in ids]
    
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
    def _predict_scores_inner(self, sentences):
        if self.model_type in ['deberta', 'tinybert']:
            inputs = self.model_tokenizer(sentences, padding=True, return_tensors ='pt').to(self.device)
            return softmax(self.model(**inputs)[0])

        t_sentences = [self.model_tokenizer.tokenize(s) for s in sentences]
        pad = max(len(s) for s in t_sentences)
        input_ids = [[101] +[self.model_tokenizer.vocab[token] for token in tokens] + [102] + [0]*(pad-len(tokens)) for tokens in t_sentences]
        res = softmax(self.model(input_ids)[0])
        return softmax(self.model(input_ids)[0])
    
    @torch.no_grad()
    def _predict_scores(self, sentences):
        batch = 500
        outputs = [self._predict_scores_inner(sentences[i: i+batch]) for i in range(0, len(sentences), batch)]
        outputs = torch.cat(outputs)
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
    def words_distributions(sentences, labels, tokenizer, num_removes = 20):
        stopwords = get_stopwords()
        c_pos = Counter()
        c_neg = Counter()
        
        for sentence, label in zip(sentences, labels):
            if label == 1:
                c_pos.update(tokenizer.tokenize(sentence))
            else:
                c_neg.update(tokenizer.tokenize(sentence))

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
        ls = legends[:]
        
        reverse = '$\mathcal{G}_{\mathsf{pr}}^{-1}$ α=0.5'
        baseline = '$\mathcal{G}_{\mathsf{b}}$'
        if reverse in ls:
            ls.remove(reverse)
            ls.remove(baseline)
            
        for i in range(len(ls)):
            self.set_tokens(pos_tokens_arr[i], neg_tokens_arr[i])
            pos_scores, neg_scores = self._aopc_global(self.pos_tokens, self.pos_sentences, 1), self._aopc_global(self.neg_tokens, self.neg_sentences, 0)
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat(str(legends[i]), num_removes+1))), columns=pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat(str(legends[i]), num_removes+1))), columns=neg_df.columns)])
        
        pos_scores, neg_scores = ScoreUtils.get_scores_dict(self.seed_path, trail_path = f"{self.delta}/scores.xlsx", alpha = self.alpha)
        pos_tokens, neg_tokens = list(pos_scores.keys()), list(neg_scores.keys())
        self.set_tokens(pos_tokens, neg_tokens)
        
        if reverse in legends:
            pos_scores, neg_scores = self._aopc_global(self.pos_tokens[::-1], self.pos_sentences, 1), self._aopc_global(self.neg_tokens[::-1], self.neg_sentences, 0)
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat(reverse, num_removes+1))), columns=pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat(reverse, num_removes+1))), columns=neg_df.columns)])
            pos_tokens_arr.append(self.pos_tokens[::-1])
            neg_tokens_arr.append(self.neg_tokens[::-1])
            
        if baseline in legends:
            p, n = self.words_distributions(self.pos_sentences+self.neg_sentences, [1]*len(self.pos_sentences)+[0]*len(self.neg_sentences), self.tokenizer)
            pos_scores, neg_scores = self._aopc_global(p, self.pos_sentences, 1), self._aopc_global(n, self.neg_sentences, 0)
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat(baseline, num_removes+1))), columns=pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat(baseline, num_removes+1))), columns=neg_df.columns)])
            pos_tokens_arr.append(p)
            neg_tokens_arr.append(n)
        
        if self.seed==42:
            self.print_tokens(legends, pos_tokens_arr, neg_tokens_arr)
            os.makedirs(f"{self.seed_path}tokens", exist_ok=True)
            for l in legends:
                pickle.dump(pos_tokens_arr[legends.index(l)], open(f"{self.seed_path}tokens/{l}_pos_tokens.pickle", "wb"))
                pickle.dump(neg_tokens_arr[legends.index(l)], open(f"{self.seed_path}tokens/{l}_neg_tokens.pickle", "wb"))
            
        return pos_df, neg_df, xlabel, ylabel, hue, legends, self.title + f' {hue}', plotter, normalizer
    
    def compare_deltas(self, **kwargs):
        deltas = [0.1, 0.15, 0.2, 0.35, 0.5]#, 0.6, 0.7, 0.8]
        get_scores_fn = lambda cur_delta: ScoreUtils.get_scores_dict(self.seed_path, trail_path = f"{cur_delta}/scores.xlsx", alpha = self.alpha)
        return self.compare_aopcs(deltas, get_scores_fn, deltas, '$\delta$', normalizer=self.delta)
    
    def compare_alphas(self, **kwargs):
        alphas = [0.95, 0.8, 0.65, 0.5]
        get_scores_fn = lambda alpha: ScoreUtils.get_scores_dict(self.seed_path, trail_path = f"{self.delta}/scores.xlsx", alpha = alpha)
        return self.compare_aopcs(alphas, get_scores_fn, alphas, '$\alpha$', normalizer=0.95)
    
    def compare_optimizations(self, **kwargs):
        optimizations = kwargs['opts'] if 'opts' in kwargs else self.opts
        get_scores_fn = lambda optimization: ScoreUtils.get_scores_dict(self.seed_path, trail_path = f"{optimization}/scores.xlsx", alpha = self.alpha)
        return self.compare_aopcs(optimizations, get_scores_fn, optimizations, 'optimization', normalizer=self.delta)
    
    def compare_aggregations(self, **kwargs):
        if 'agg_params' not in kwargs:
            legends = ['$\mathcal{G}_{\mathsf{sq}}$', '$\mathcal{G}_{\mathsf{h}}$', '$\mathcal{G}_{\mathsf{pr}}$ α=0.5', '$\mathcal{G}_{\mathsf{pr}}$ α=0.95', '$\mathcal{G}_{\mathsf{av}}$', '$\mathcal{G}^{+5}_{\mathsf{av}}$']
            aggregations = ['root_', 'homogenity_', '', '', 'avg_', f'avg_{3e-3}_']
            alphas = [None, None, 0.5, 0.95, None, None]
            
        else:
            aggregations, alphas, legends = kwargs['agg_params']
        get_scores_fn = lambda x: ScoreUtils.get_scores_dict(self.seed_path, trail_path=f"{self.delta}/{x[0]}scores.xlsx", alpha = x[1])
        
        legends += ['$\mathcal{G}_{\mathsf{pr}}^{-1}$ α=0.5', '$\mathcal{G}_{\mathsf{b}}$']
        
        return self.compare_aopcs(zip(aggregations, alphas), get_scores_fn, legends, 'aggregation', normalizer='$\mathcal{G}_{\mathsf{pr}}$ α=0.5')
        
    def compare_percents(self, **kwargs):
        percents = [5, 10, 25, 50, 75, 100]
        get_scores_fn = lambda percent: ScoreUtils.get_scores_dict(self.seed_path, trail_path = f"{self.delta}/percents/scores-{percent}.xlsx", alpha = self.alpha)
        return self.compare_aopcs(percents, get_scores_fn, percents, 'percent', normalizer=100)

    def compare_percents_remove(self, **kwargs):
        percents = [5, 10, 25, 50, 75, 100]
        get_scores_fn = lambda percent: ScoreUtils.get_scores_dict(self.seed_path, trail_path = f"{self.delta}/percents/scores-{percent}.xlsx", alpha = self.alpha)
        return self.compare_aopcs(percents, get_scores_fn, percents, 'percent', plotter=AOPC_Plotter.time_aopc_plot, normalizer=100)
          
    def time_percent(self, **kwargs):
        opts = kwargs['opts'] if 'opts' in kwargs else self.opts
        return self.time_percent_monitor(opts, alpha = self.alpha)
        
    def time_aopc(self, **kwargs):
        opts = kwargs['opts'] if 'opts' in kwargs else self.opts
        return self.time_aopc_monitor(opts, alpha = self.alpha)
    
    def time_aopc_aggregation(self, **kwargs):
        if 'agg_params' not in kwargs:
            legends = ['$\mathcal{G}_{\mathsf{sq}}$', '$\mathcal{G}_{\mathsf{h}}$', '$\mathcal{G}_{\mathsf{pr}}$ α=0.5', '$\mathcal{G}_{\mathsf{pr}}$ α=0.95', '$\mathcal{G}_{\mathsf{av}}$', '$\mathcal{G}^{+5}_{\mathsf{av}}$']
            aggregations = ['root_', 'homogenity_', '', '', 'avg_', f'avg_{3e-3}_']
            alphas = [None, None, 0.5, 0.95, None, None]
            
        else:
            legends, aggregations, alphas= kwargs['agg_params']
        return self.time_aopc_aggregation_monitor(aggregations, alphas, legends)
    
    @staticmethod
    def plot_all_aopcs(model_type, opt_prefix = 'stop-words-', c='aopc time', datasets = ['corona', 'toy-spam', 'home-spam', 'dilemma'], seeds=[42, 84, 126, 168, 210], alone = False, **kwargs):
        title_dict = {'aopc time': 'Optimizations', "aggregation": "Aggregations", 'percents time': 'Optimizations Percents'}
        normalizer_dict = {'aopc time': '0.1', "aggregation":  '$\mathcal{G}_{\mathsf{pr}}$ α=0.5', 'percents time': '0.1'}
        normalizer = normalizer_dict[c]
        title = title_dict[c]
        pos_dfs, neg_dfs = [], []
        plotter = AOPC_Plotter.aopc_plot_agg if c == 'aggregation' else AOPC_Plotter.aopc_plot_opt
        
        for ds in datasets:
            path = f'results/mp/{model_type}/{ds}/confidence/'
            pos_df = pd.read_csv(f'{path}/{opt_prefix}{c}_pos_aopc.csv')
            neg_df = pd.read_csv(f'{path}/{opt_prefix}{c}_neg_aopc.csv')
            if 'Unnamed: 0' in pos_df.columns:
                pos_df.drop(columns = ['Unnamed: 0'], inplace = True)
                neg_df.drop(columns = ['Unnamed: 0'], inplace = True)
            if "# of features removed" in pos_df.columns:
                pos_df = pos_df[pos_df["# of features removed"]<=20]
                neg_df = neg_df[neg_df["# of features removed"]<=20]

            legends = list(pos_df.iloc[:, -1].unique())
            xlabel, ylabel, hue = pos_df.columns
 
            jump = len(pos_df)//len(seeds)//len(legends)
            jumps = list(range(jump-1, len(pos_df)//len(legends), jump))
            pos_normalizer = pos_df[pos_df[hue].astype('str')==str(normalizer)].iloc[jumps, 1].mean()
            jump = len(neg_df)//len(seeds)//len(legends)
            jumps = list(range(jump-1, len(neg_df)//len(legends), jump))
            neg_normalizer = neg_df[neg_df[hue].astype('str')==str(normalizer)].iloc[jumps, 1].mean()
            pos_df.iloc[:, 1]/=pos_normalizer
            neg_df.iloc[:, 1]/=neg_normalizer
            
            pos_dfs.append(pos_df); neg_dfs.append(neg_df)
            
        if alone:
            plotter(pos_dfs, neg_dfs, xlabel, ylabel, hue, legends, title, datasets, values = ['0.1', 'desired-0.1', 'topk-0.1', 'masking-0.1', 'topk-desired-masking-0.1'])
    
        else: 
            plotter(pos_dfs, neg_dfs, xlabel, ylabel, hue, legends, title, datasets)
            
    @staticmethod
    def plot_single(model_type, opt_prefix = 'stop-words-', c='aopc time', datasets = ['corona', 'toy-spam', 'home-spam', 'dilemma'], seeds=[42, 84, 126, 168, 210], alone = False, df_type = 'pos', **kwargs):
        title_dict = {'aopc time': 'Optimizations', "aggregation": "Aggregations", 'percents time': 'Optimizations Percents'}
        normalizer_dict = {'aopc time': '0.1', "aggregation":  '$\mathcal{G}_{\mathsf{pr}}$ α=0.5', 'percents time': '0.1'}
        normalizer = normalizer_dict[c]
        title = title_dict[c]
        pos_dfs, neg_dfs = [], []
        plotter = AOPC_Plotter.plot_single
        
        for ds in datasets:
            path = f'results/mp/{model_type}/{ds}/confidence/'
            pos_df = pd.read_csv(f'{path}/{opt_prefix}{c}_pos_aopc.csv')
            neg_df = pd.read_csv(f'{path}/{opt_prefix}{c}_neg_aopc.csv')
            if 'Unnamed: 0' in pos_df.columns:
                pos_df.drop(columns = ['Unnamed: 0'], inplace = True)
                neg_df.drop(columns = ['Unnamed: 0'], inplace = True)
            if "# of features removed" in pos_df.columns:
                pos_df = pos_df[pos_df["# of features removed"]<=20]
                neg_df = neg_df[neg_df["# of features removed"]<=20]

            legends = list(pos_df.iloc[:, -1].unique())
            xlabel, ylabel, hue = pos_df.columns
 
            jump = len(pos_df)//len(seeds)//len(legends)
            jumps = list(range(jump-1, len(pos_df)//len(legends), jump))
            pos_normalizer = pos_df[pos_df[hue].astype('str')==str(normalizer)].iloc[jumps, 1].mean()
            jump = len(neg_df)//len(seeds)//len(legends)
            jumps = list(range(jump-1, len(neg_df)//len(legends), jump))
            #neg_normalizer = neg_df[neg_df[hue].astype('str')==str(normalizer)].iloc[jumps, 1].mean()
            neg_normalizer =1
            pos_df.iloc[:, 1]/=pos_normalizer
            neg_df.iloc[:, 1]/=neg_normalizer
            
            pos_dfs.append(pos_df); neg_dfs.append(neg_df)
        
        dfs = pos_dfs if df_type == 'pos' else neg_dfs
        plotter(dfs, xlabel, ylabel, hue, legends, title, datasets)
    
    def compare_all(self, seeds=[42, 84, 126, 168, 210], from_img=[], skip=[], only=None, verbose= True, **kwargs): 
        if not verbose:
            global print
            print = lambda *x: None
        compares = {'delta': self.compare_deltas, 'alpha': self.compare_alphas, 'aggregation': self.compare_aggregations, 'percent': self.compare_percents,  'optimization': self.compare_optimizations, 'percents time': self.time_percent, 'aopc time': self.time_aopc, 'aggregation-aopc time': self.time_aopc_aggregation}
        
        normalizers = dict(zip(compares.keys(),[0.1, 0.95, '$\mathcal{G}_{\mathsf{pr}}$ α=0.5', 100, self.delta, self.delta, self.delta, '$\mathcal{G}_{\mathsf{pr}}$ α=0.5']))
        
        for c in compares:
            if only and c not in only:
                continue
            if c in skip:
                continue
            elif c in from_img:
                pos_df = pd.read_csv(f'{self.path}/{self.opt_prefix}{c}_pos_aopc.csv')
                neg_df = pd.read_csv(f'{self.path}/{self.opt_prefix}{c}_neg_aopc.csv')
                if 'Unnamed: 0' in pos_df.columns:
                    pos_df.drop(columns = ['Unnamed: 0'], inplace = True)
                    neg_df.drop(columns = ['Unnamed: 0'], inplace = True)
                if "# of features removed" in pos_df.columns:
                    pos_df = pos_df[pos_df["# of features removed"]<=20]
                    neg_df = neg_df[neg_df["# of features removed"]<=20]
                
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

                pos_tok_arr = [pickle.load(open(f"{self.path}42/{self.opt_prefix}tokens/{l}_pos_tokens.pickle", "rb")) for l in legends]
                neg_tok_arr = [pickle.load(open(f"{self.path}42/{self.opt_prefix}tokens/{l}_neg_tokens.pickle", "rb")) for l in legends]
                self.print_tokens(legends, pos_tok_arr, neg_tok_arr)
                
                limiter = c in ['percents time', 'aopc time']
                if c in ['percents time', 'aopc time', 'aggregation-aopc time']:
                    c_title = self.title + f' {c.split()[0]}'
                if c in ['optimization', 'percents time', 'aopc time']:
                    plotter = AOPC_Plotter.aopc_plot_time
                else:
                    plotter = AOPC_Plotter.aopc_plot
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
                    
                limiter = c in ['percents time', 'aopc time']
                if c in ['optimization', 'percents time', 'aopc time']:
                    plotter = AOPC_Plotter.aopc_plot_time
                plotter(pos_df, neg_df, xlabel, ylabel, hue, legends, c_title, limiter)

                
    def time_percent_monitor(self, opts, alpha=0.95):
        """
        compare top k anchors during runtime to the final top k of the default running
        """
        top = self.num_removes
        get_exps = lambda opt: pickle.load(open(f"{self.seed_path}{opt}/exps_list.pickle", "rb"))
        times = pd.read_csv('times.csv', index_col=0)
        pos_percent = sum(self.labels)/len(self.labels)
        default_res = ScoreUtils.calculate_time_scores(self.tokenizer, self.sentences, get_exps(self.delta), self.predictions, [alpha])[alpha]
        final_top_pos = set(default_res['pos'][100].index[:top])
        final_top_neg = set(default_res['neg'][100].index[:top])
        pos_df = pd.DataFrame(columns = ['time (minutes)', "percents", "optimization"])
        neg_df = pd.DataFrame(columns = pos_df.columns)

        for opt in opts:
            percents_dict = ScoreUtils.calculate_time_scores(self.tokenizer, self.sentences, get_exps(opt), self.predictions, [alpha])[alpha]
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
            
        return pos_df, neg_df, 'time (minutes)', 'percents', "optimization", opts, f'{self.ds_name} dataset percents', AOPC_Plotter.aopc_plot, self.delta
       
    def time_aopc_aggregation_monitor(self, aggregations, alphas, legends):
        """
        compare best topk negative and positive anchors between current result and the
        end result top scores for different aggregations
        """
        top = self.num_removes
        exps = pickle.load(open(f"{self.seed_path}0.1/exps_list.pickle", "rb"))
        times = pd.read_csv('times.csv', index_col=0)
        pos_percent = sum(self.labels)/len(self.labels)
        pos_df = pd.DataFrame(columns = ['time (minutes)', "AOPC-global", "aggregation"])
        neg_df = pd.DataFrame(columns = pos_df.columns)
        pos_tok_arr, neg_tok_arr = [], []

        for agg, alpha, legend in zip(aggregations, alphas, legends):
            pos_results, neg_results = [], []
            if agg=='':
                percents_dict = ScoreUtils.calculate_time_scores(self.tokenizer, self.sentences, exps, self.predictions, [alpha])[alpha]
            else:
                splitted_agg = agg.split('_')
                agg_name = splitted_agg[0]
                min_count = float(splitted_agg[1]) if (agg_name=='avg' and len(splitted_agg[1])>0) else 0
                percents_dict = ScoreUtils.calculate_time_aggregation(self.tokenizer, self.sentences, exps, self.predictions, agg_name, min_count)
            percents = percents_dict['pos'].keys()
            for i in percents:
                top_pos = list(percents_dict['pos'][i].index[:top])
                top_neg = list(percents_dict['neg'][i].index[:top])
                self.set_tokens(top_pos, top_neg)
                pos_results.append(2*self._aopc_global(top_pos, self.pos_sentences, 1, [0, top])[1])
                neg_results.append(2*self._aopc_global(top_neg, self.neg_sentences, 0, [0, top])[1])
            pos_tok_arr.append(top_pos)
            neg_tok_arr.append(top_neg)
                
            time = times.loc[f'mp/{self.model_type}/{self.ds_name}/confidence/42/{self.opt_prefix}0.1'].time
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip([time*pos_percent*i/100 for i in percents], pos_results, np.repeat(legend, len(pos_results)))), columns = pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip([time*(1-pos_percent)*i/100 for i in percents], neg_results, np.repeat(legend, len(neg_results)))), columns = neg_df.columns)])

        if self.seed==42:
            self.print_tokens(legends, pos_tok_arr, neg_tok_arr)
                
        return pos_df, neg_df, 'time (minutes)', 'AOPC-global', "aggregation", legends, f'{self.ds_name} dataset time aggregation', AOPC_Plotter.aopc_plot, '$\mathcal{G}_{\mathsf{pr}}$ α=0.5'
    

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
            percents_dict = ScoreUtils.calculate_time_scores(self.tokenizer, self.sentences, get_exps(opt), self.predictions, [alpha])[alpha]
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
                
        return pos_df, neg_df, 'time (minutes)', 'AOPC-global', "optimization", opts, f'{self.ds_name} dataset aopc', AOPC_Plotter.aopc_plot, self.delta
    
    @staticmethod
    def print_tokens(legends, pos_tokens_arr, neg_tokens_arr):
        print('pos')
        for l in legends:
            print(colors[legends.index(l)](f'{l}:'), f'{pos_tokens_arr[legends.index(l)][:10]}')
        print('\nneg')
        for l in legends:
            print(colors[legends.index(l)](f'{l}:'), f'{neg_tokens_arr[legends.index(l)][:10]}')
