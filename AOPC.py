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
from torch.nn.functional import softmax
from myUtils import set_seed
from score import ScoreUtils

class AOPC_Plotter:
    @staticmethod
    def aopc_plot(pos_y_arr, neg_y_arr, legends, graph_title, pos_x_arr = None, neg_x_arr = None, xlabel = '# of features removed', ylabel = 'AOPC - global'):
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        fig.supxlabel(xlabel, x=0.51)
        fig.supylabel(ylabel, x=0.08)
        fig.suptitle(graph_title)
        
        y_arrs = [pos_y_arr, neg_y_arr]
        if pos_x_arr is None:
            pos_x_arr = np.tile(np.arange(len(pos_y_arr[0])), (len(pos_y_arr), 1))
            neg_x_arr = np.tile(np.arange(len(neg_y_arr[0])), (len(neg_y_arr), 1))
        x_arrs = [pos_x_arr, neg_x_arr]
        ax_titles = ['positive', 'negative']

        for i in range(len(axs)):
            for ys, xs, legend in zip(y_arrs[i], x_arrs[i], legends):
                axs[i].plot(xs, ys, label = legend)
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
    @staticmethod
    def init(model, tokenizer, sentences, labels, pos_tokens, neg_tokens, title, num_removes = 30):
        AOPC.model = model
        AOPC.tokenizer = tokenizer
        AOPC.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        AOPC.tokens_method = AOPC._remove_tokens
        AOPC.title = title
        AOPC.num_removes = num_removes
        AOPC.pos_tokens, AOPC.neg_tokens = pos_tokens, neg_tokens
        AOPC.pos_sentences = [tokenizer.tokenize(s) for i, s in enumerate(sentences) if labels[i]==1]
        AOPC.neg_sentences = [tokenizer.tokenize(s) for i, s in enumerate(sentences) if labels[i]==0]
        
    @staticmethod
    def _remove_tokens(idx, tokens, sentences):
        return [['[PAD]' if token in tokens[:idx] else token for token in sentence] for sentence in sentences]
    
    @staticmethod
    def _replace_tokens(idx, tokens, sentences):
        unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
        replaced_sentences = []
        for s in sentences:
            tokenized_sentence = copy.deepcopy(s)
            for i in range(len(tokenized_sentence)):
                token = tokenized_sentence[i]
                if token in tokens[:idx]:
                    tokenized_sentence[i] = '[MASK]'
                    sentence = AOPC.tokenizer.decode(AOPC.tokenizer.encode(tokenized_sentence))
                    results = unmasker(sentence, top_k=2)
                    for r in results:
                        if r['token_str']!=token:
                            tokenized_sentence[i] = r['token_str']
                            break

            replaced_sentences.append(tokenized_sentence)
        return replaced_sentences      

    @staticmethod
    def _predict_scores(sentences):
        pad = max(len(s) for s in sentences)
        input_ids = [[101] +[AOPC.tokenizer.vocab[token] for token in tokens] + [102] + [0]*(pad-len(tokens)) for tokens in sentences]
        input_ids = torch.tensor(input_ids, device=AOPC.device)
        attention_mask = [[1]*(len(tokens)+2)+[0]*(pad-len(tokens)) for tokens in sentences]
        attention_mask = torch.tensor(attention_mask, device=AOPC.device)
        outputs = softmax(AOPC.model(input_ids = input_ids, attention_mask=attention_mask)[0])
        return outputs.detach().cpu().numpy()
    
    @staticmethod
    def _aopc_predictions(sentences_arr, label):
        return np.array([AOPC._predict_scores(sentences)[:, label] for sentences in sentences_arr])
    
    
    # @staticmethod
    # def _aopc_predictions(sentences_arr, label):
    #     predictions = []
    #     for sentences in sentences_arr:
    #         predictions_batch = []
    #         for i in range(0, len(sentences), 50):
    #             predictions_batch.extend(AOPC._predict_scores(sentences[i:i+50])[:, label])
    #         predictions.append(predictions_batch)
    #     return np.array(predictions)
    
    @staticmethod
    def _aopc_formula(orig_predictions, predictions_arr, k):
        N = len(orig_predictions)    
        return sum([sum(orig_predictions - predictions_arr[i])/N for i in range(k+1)])/(k+1)

    @staticmethod
    def _calc_aopc(predictions_arr):
        orig_predictions = predictions_arr[0]
        return [AOPC._aopc_formula(orig_predictions, predictions_arr, k) for k in range(len(predictions_arr))]

    @staticmethod
    def _aopc_global(tokens, sentences, label, removes_arr = None):
        if not removes_arr:
            removes_arr = range(AOPC.num_removes+1)
        removed_sentences_arr = []
        removed_sentences = sentences
        for i in removes_arr:
            removed_sentences = AOPC.tokens_method(i, tokens, removed_sentences)
            removed_sentences_arr.append(removed_sentences) 
        predictions_arr = AOPC._aopc_predictions(removed_sentences_arr, label)
        aopc_scores = AOPC._calc_aopc(predictions_arr)
        return aopc_scores  
        
    @staticmethod
    def _compare_sorts(tokens_method = 'remove', legends = ['normal', 'random', 'reverse', 'baseline'], plotter = AOPC_Plotter.aopc_plot):
        AOPC.tokens_method = AOPC._remove_tokens if tokens_method=='remove' else AOPC._replace_tokens
        pos_scores, neg_scores = [], []
        if 'normal' in legends:
            pos_scores.append(AOPC._aopc_global(AOPC.pos_tokens, AOPC.pos_sentences, 1))
            neg_scores.append(AOPC._aopc_global(AOPC.neg_tokens, AOPC.neg_sentences, 0))
        
        if 'random' in legends:
            random_scores = np.zeros(AOPC.num_removes+1)
            for i in range(5):
                set_seed((i+1)*100)
                shuffled_tokens = AOPC.pos_tokens.copy()
                random.shuffle(shuffled_tokens)
                random_scores += np.array(AOPC._aopc_global(shuffled_tokens, AOPC.pos_sentences, 1))
            random_scores/=5
            pos_scores.append(random_scores)
            
            random_scores = np.zeros(AOPC.num_removes+1)
            for i in range(5):
                set_seed((i+1)*100)
                shuffled_tokens = AOPC.neg_tokens.copy()
                random.shuffle(shuffled_tokens)
                random_scores += np.array(AOPC._aopc_global(shuffled_tokens, AOPC.neg_sentences, 0))
            random_scores/=5
            neg_scores.append(random_scores)
        
        if 'reverse' in legends:
            pos_scores.append(AOPC._aopc_global(AOPC.pos_tokens[::-1], AOPC.pos_sentences, 1))
            neg_scores.append(AOPC._aopc_global(AOPC.neg_tokens[::-1], AOPC.neg_sentences, 0))
            
        if 'baseline' in legends:
            p, n = AOPC.words_distributions(AOPC.pos_sentences+AOPC.neg_sentences, [1]*len(AOPC.pos_sentences)+[0]*len(AOPC.neg_sentences), AOPC.tokenizer)
            pos_scores.append(AOPC._aopc_global(p, AOPC.pos_sentences, 1))
            neg_scores.append(AOPC._aopc_global(n, AOPC.neg_sentences, 0))

        plotter(pos_scores, neg_scores, legends, AOPC.title)
        
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
            AOPC.init(model, tokenizer, sentences, labels, pos_tokens_arr[i], neg_tokens_arr[i], title, num_removes = num_removes) 
            pos_scores.append(AOPC._aopc_global(AOPC.pos_tokens, AOPC.pos_sentences, 1))
            neg_scores.append(AOPC._aopc_global(AOPC.neg_tokens, AOPC.neg_sentences, 0))
        
        plotter(pos_scores, neg_scores, legends, title)
        
    @staticmethod
    def compare_all(folder_name, model, tokenizer, sentences, labels, title, num_removes = 30, from_img = [], skip = []):
        def compare_sorts():
            pos_scores, neg_scores = ScoreUtils.get_scores_dict(folder_name, trail_path = "../0.1/scores.xlsx")
            pos_tokens, neg_tokens = list(pos_scores.keys()), list(neg_scores.keys())
            AOPC.init(model, tokenizer, sentences, labels, pos_tokens, neg_tokens, title + ' sorts', num_removes)
            AOPC._compare_sorts()
            
        def compare_deltas():
            deltas = [0.1, 0.15, 0.2, 0.35, 0.5, 0.6, 0.7]
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
                AOPC.init(model, tokenizer, sentences, labels, pos_tokens_arr[i], neg_tokens_arr[i], title, num_removes = num_removes) 
                pos_result = AOPC._aopc_global(AOPC.pos_tokens, AOPC.pos_sentences, 1)
                neg_result = AOPC._aopc_global(AOPC.neg_tokens, AOPC.neg_sentences, 0)
            
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
        fig.savefig(f'results/graphs/{title}', bbox_inches='tight')
            
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
        times = pd.read_csv('times.csv', index_col=0)
        pos_percent = sum(labels)/len(labels)
        default_res = ScoreUtils.calculate_time_scores(tokenizer, sentences, get_exps('0.1'), labels,[alpha])[alpha]
        final_top_pos = set(default_res['pos'][100].index[:top])
        final_top_neg = set(default_res['neg'][100].index[:top])
        pos_df = pd.DataFrame(columns = ['time (minutes)', "percents", "optimization"])
        neg_df = pd.DataFrame(columns = pos_df.columns)

        for opt in opts:
            percents_dict = ScoreUtils.calculate_time_scores(tokenizer,sentences,get_exps(opt),labels,[alpha])[alpha]
            pos_results, neg_results = [], []
            percents = percents_dict['pos'].keys()
            for i in percents:
                top_pos = set(percents_dict['pos'][i].index[:top])
                pos_results.append(len(top_pos.intersection(final_top_pos))/top)
                top_neg = set(percents_dict['neg'][i].index[:top])
                neg_results.append(len(top_neg.intersection(final_top_neg))/top)
            try:
                time = times.loc[f'{model_type}/{ds_name}/confidence/{opt}'].time
            except:
                time = times.loc[f'{model_type}/{ds_name}/confidence/{opt}-0.1'].time
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip([time*pos_percent*i/100 for i in percents], pos_results, np.repeat(opt, len(pos_results)))), columns = pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip([time*(1-pos_percent)*i/100 for i in percents], neg_results, np.repeat(opt, len(neg_results)))), columns = neg_df.columns)])
            
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        sns.lineplot(data=pos_df, x='time (minutes)', y="percents", hue = "optimization", legend = opts, ax = axs[0], palette=sns.color_palette()).set(title="positive")
        sns.lineplot(data=neg_df, x='time (minutes)', y="percents", hue = "optimization", legend = opts, ax = axs[1], palette=sns.color_palette()).set(title="negative")
        fig.suptitle(f'{model_type} {ds_name} percents time')
        plt.show()
        fig.savefig(f'results/graphs/{model_type} {ds_name} percents time', bbox_inches='tight')
        #AOPC_Plotter.aopc_plot(plot_dict['pos_y'], plot_dict['neg_y'], opts, f'{model_type} {ds_name} percentage time', plot_dict['pos_x'], plot_dict['neg_x'], 'time (minutes)', 'percents')
                
    @staticmethod
    def time_aopc_monitor(path, title, model, model_type, ds_name, opts, tokenizer, sentences, labels, top=30, alpha = 0.95):
        """
        compare best topk negative and positive anchors between current result and the
        end result top scores
        """
        get_exps = lambda opt: pickle.load(open(f"{path}/../{opt}/exps_list.pickle", "rb"))
        times = pd.read_csv('times.csv', index_col=0)
        pos_percent = sum(labels)/len(labels)
        plot_dict = defaultdict(list)

        for opt in opts:
            pos_scores, neg_scores = [], []
            percents_dict = ScoreUtils.calculate_time_scores(tokenizer,sentences,get_exps(opt),labels,[alpha])[alpha]
            percents = percents_dict['pos'].keys()
            for i in percents:
                top_pos = list(percents_dict['pos'][i].index[:top])
                top_neg = list(percents_dict['neg'][i].index[:top])
                AOPC.init(model, tokenizer, sentences, labels,top_pos,top_neg,title,num_removes=top)
                pos_scores.append(2*AOPC._aopc_global(top_pos, AOPC.pos_sentences, 1, [0, top])[1])
                neg_scores.append(2*AOPC._aopc_global(top_neg, AOPC.neg_sentences, 0, [0, top])[1])
            
            try:
                time = times.loc[f'{model_type}/{ds_name}/confidence/{opt}'].time
            except:
                time = times.loc[f'{model_type}/{ds_name}/confidence/{opt}-0.1'].time
            plot_dict['pos_x'].append([time*pos_percent*i/100 for i in percents])
            plot_dict['neg_x'].append([time*(1-pos_percent)*i/100 for i in percents])
            plot_dict['pos_y'].append(pos_scores)
            plot_dict['neg_y'].append(neg_scores)

        AOPC_Plotter.aopc_plot(plot_dict['pos_y'], plot_dict['neg_y'], opts, f'{model_type} {ds_name} aopc time', plot_dict['pos_x'], plot_dict['neg_x'], 'time (minutes)')