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
    def aopc_plot(pos_df, neg_df, xlabel, ylabel, hue, legend, title):
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        sns.lineplot(data=pos_df, x=xlabel, y=ylabel, hue=hue, legend=legend, ax = axs[0], palette=sns.color_palette()).set(title="positive")
        sns.lineplot(data=neg_df, x=xlabel, y=ylabel, hue=hue, legend=legend, ax = axs[1], palette=sns.color_palette()).set(title="negative")
        fig.suptitle(title)
        plt.show()
        fig.savefig(f'results/graphs/{title}', bbox_inches='tight')
        
    @staticmethod       
    def time_aopc_plot(pos_df, neg_df, xlabel, ylabel, hue, legend, title):
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        pos_df = pos_df[pos_df[xlabel].isin([1, 5 ,10, 20 ,30])]
        neg_df = neg_df[neg_df[xlabel].isin([1, 5 ,10, 20 ,30])]
        sns.lineplot(data=pos_df, x=hue, y=ylabel, hue=xlabel, legend=legend, ax = axs[0], palette=sns.color_palette()).set(title="positive")
        sns.lineplot(data=neg_df, x=hue, y=ylabel, hue=xlabel, legend=legend, ax = axs[1], palette=sns.color_palette()).set(title="negative")
        fig.suptitle(title)
        sns.move_legend(axs[0], "lower left")
        sns.move_legend(axs[1], "lower left")
        plt.show()
        fig.savefig(f'results/graphs/{title}', bbox_inches='tight')

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
    def _compare_sorts(tokens_method = 'remove', legends = ['normal', 'random', 'reverse', 'baseline'], num_removes = 30, hue = 'sorts', xlabel="# of features removed", ylabel="AOPC-global", plotter = AOPC_Plotter.aopc_plot):
        AOPC.tokens_method = AOPC._remove_tokens if tokens_method=='remove' else AOPC._replace_tokens
        pos_df = pd.DataFrame(columns = [xlabel, ylabel, hue])
        neg_df = pd.DataFrame(columns = pos_df.columns)
        
        if 'normal' in legends:
            pos_scores, neg_scores = AOPC._aopc_global(AOPC.pos_tokens, AOPC.pos_sentences, 1), AOPC._aopc_global(AOPC.neg_tokens, AOPC.neg_sentences, 0)
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat('normal', num_removes+1))), columns=pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat('normal', num_removes+1))), columns=neg_df.columns)])

        if 'random' in legends:
            for i in range(5):
                set_seed((i+1)*100)
                shuffled_pos, shuffled_neg = AOPC.pos_tokens.copy(), AOPC.neg_tokens.copy()
                random.shuffle(shuffled_pos), random.shuffle(shuffled_neg)
                pos_scores, neg_scores = AOPC._aopc_global(shuffled_pos, AOPC.pos_sentences, 1), AOPC._aopc_global(shuffled_neg, AOPC.neg_sentences, 0)
                pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat('random', num_removes+1))), columns=pos_df.columns)])
                neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat('random', num_removes+1))), columns=pos_df.columns)])

        if 'reverse' in legends:
            pos_scores, neg_scores = AOPC._aopc_global(AOPC.pos_tokens[::-1], AOPC.pos_sentences, 1), AOPC._aopc_global(AOPC.neg_tokens[::-1], AOPC.neg_sentences, 0)
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat('reverse', num_removes+1))), columns=pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat('reverse', num_removes+1))), columns=neg_df.columns)])

        if 'baseline' in legends:
            p, n = AOPC.words_distributions(AOPC.pos_sentences+AOPC.neg_sentences, [1]*len(AOPC.pos_sentences)+[0]*len(AOPC.neg_sentences), AOPC.tokenizer)
            pos_scores, neg_scores = AOPC._aopc_global(p, AOPC.pos_sentences, 1), AOPC._aopc_global(n, AOPC.neg_sentences, 0)
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat('baseline', num_removes+1))), columns=pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat('baseline', num_removes+1))), columns=neg_df.columns)])

        plotter(pos_df, neg_df, xlabel, ylabel, hue, legends, AOPC.title + f' {hue}')
        
    @staticmethod
    def compare_aopcs(model, tokenizer, compare_list, get_scores_fn, sentences, labels, legends, title, num_removes, hue, xlabel="# of features removed", ylabel="AOPC-global", plotter=AOPC_Plotter.aopc_plot):         
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
            AOPC.init(model, tokenizer, sentences, labels, pos_tokens_arr[i], neg_tokens_arr[i], title, num_removes = num_removes)
            pos_scores, neg_scores = AOPC._aopc_global(AOPC.pos_tokens, AOPC.pos_sentences, 1), AOPC._aopc_global(AOPC.neg_tokens, AOPC.neg_sentences, 0)
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip(np.arange(num_removes+1), pos_scores, np.repeat(legends[i], num_removes+1))), columns=pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip(np.arange(num_removes+1), neg_scores, np.repeat(legends[i], num_removes+1))), columns=neg_df.columns)])
        
        plotter(pos_df, neg_df, xlabel, ylabel, hue, legends, title + f' {hue}')
        
    @staticmethod
    def compare_all(folder_name, model, tokenizer, sentences, labels, title, num_removes = 30, from_img = [], skip = []):
        
        def compare_sorts():
            pos_scores, neg_scores = ScoreUtils.get_scores_dict(folder_name, trail_path = "../0.1/scores.xlsx")
            pos_tokens, neg_tokens = list(pos_scores.keys()), list(neg_scores.keys())
            AOPC.init(model, tokenizer, sentences, labels, pos_tokens, neg_tokens, title, num_removes)
            AOPC._compare_sorts()
            
        def compare_deltas():
            deltas = [0.1, 0.15, 0.2, 0.35, 0.5, 0.6, 0.7]
            get_scores_fn = lambda delta: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../{delta}/scores.xlsx")
            AOPC.compare_aopcs(model, tokenizer, deltas, get_scores_fn, sentences, labels, deltas, title, num_removes, 'delta')
        
        def compare_alphas():
            alphas = [0.95, 0.8, 0.65, 0.5]
            get_scores_fn = lambda alpha: ScoreUtils.get_scores_dict(folder_name, trail_path = "../0.1/scores.xlsx", alpha = alpha)
            AOPC.compare_aopcs(model, tokenizer, alphas, get_scores_fn, sentences, labels, alphas, title, num_removes, 'alpha')
        
        def compare_optimizations():
            optimizations = [str(0.1), 'lossy', 'topk', 'desired']
            get_scores_fn = lambda optimization: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../{optimization}/scores.xlsx")
            AOPC.compare_aopcs(model, tokenizer, optimizations, get_scores_fn, sentences, labels, optimizations, title, num_removes, 'optimization')
    
        def compare_aggregations():
            aggregations = ['', '', 'sum_', 'avg_']
            alphas = [0.5, 0.95, None, None]
            legends = ['probabilistic α=0.5', 'probabilistic α=0.95', 'sum', 'avg']
            get_scores_fn = lambda x: ScoreUtils.get_scores_dict(folder_name, folder_name, trail_path = f"../0.1/{x[0]}scores.xlsx", alpha = x[1])
            AOPC.compare_aopcs(model, tokenizer, zip(aggregations, alphas), get_scores_fn, sentences, labels, legends, title, num_removes, 'aggregation')
        
        def compare_percents():
            percents = [10, 25, 50, 75, 100]
            get_scores_fn = lambda percent: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../0.1/percents/scores-{percent}.xlsx", alpha = 0.95)
            AOPC.compare_aopcs(model, tokenizer, percents, get_scores_fn, sentences, labels, percents, title, num_removes, 'percent')
        
        def compare_time_percents():
            percents = [10, 25, 50, 75, 100]
            get_scores_fn = lambda percent: ScoreUtils.get_scores_dict(folder_name, trail_path = f"../0.1/percents/scores-{percent}.xlsx", alpha = 0.95)
            AOPC.compare_aopcs(model, tokenizer, percents, get_scores_fn, sentences, labels, percents, title, num_removes, 'time-percent', plotter=AOPC_Plotter.time_aopc_plot)
            
        compares = {'sorts': compare_sorts, 'deltas': compare_deltas, 'alphas': compare_alphas, 'optimizations': compare_optimizations, 'aggregations': compare_aggregations, 'percents': compare_percents, 'time-percents': compare_time_percents} 

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
            
                pos_df = pos_df.append(pd.DataFrame(zip(np.arange(num_removes+1), pos_result, np.repeat(legends[i], num_removes+1)), pos_df.columns))
                neg_df = neg_df.append(pd.DataFrame(zip(np.arange(num_removes+1), neg_result, np.repeat(legends[i], num_removes+1)), neg_df.columns))
            
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
            
        AOPC_Plotter.aopc_plot(pos_df, neg_df, 'time (minutes)', 'percents', "optimization", opts, f'{model_type} {ds_name} percents time')
                
    @staticmethod
    def time_aopc_monitor(path, title, model, model_type, ds_name, opts, tokenizer, sentences, labels, top=30, alpha = 0.95):
        """
        compare best topk negative and positive anchors between current result and the
        end result top scores
        """
        get_exps = lambda opt: pickle.load(open(f"{path}/../{opt}/exps_list.pickle", "rb"))
        times = pd.read_csv('times.csv', index_col=0)
        pos_percent = sum(labels)/len(labels)
        pos_df = pd.DataFrame(columns = ['time (minutes)', "AOPC - global", "optimization"])
        neg_df = pd.DataFrame(columns = pos_df.columns)

        for opt in opts:
            pos_results, neg_results = [], []
            percents_dict = ScoreUtils.calculate_time_scores(tokenizer,sentences,get_exps(opt),labels,[alpha])[alpha]
            percents = percents_dict['pos'].keys()
            for i in percents:
                top_pos = list(percents_dict['pos'][i].index[:top])
                top_neg = list(percents_dict['neg'][i].index[:top])
                AOPC.init(model, tokenizer, sentences, labels,top_pos,top_neg,title,num_removes=top)
                pos_results.append(2*AOPC._aopc_global(top_pos, AOPC.pos_sentences, 1, [0, top])[1])
                neg_results.append(2*AOPC._aopc_global(top_neg, AOPC.neg_sentences, 0, [0, top])[1])
            
            try:
                time = times.loc[f'{model_type}/{ds_name}/confidence/{opt}'].time
            except:
                time = times.loc[f'{model_type}/{ds_name}/confidence/{opt}-0.1'].time
            pos_df = pd.concat([pos_df, pd.DataFrame(list(zip([time*pos_percent*i/100 for i in percents], pos_results, np.repeat(opt, len(pos_results)))), columns = pos_df.columns)])
            neg_df = pd.concat([neg_df, pd.DataFrame(list(zip([time*(1-pos_percent)*i/100 for i in percents], neg_results, np.repeat(opt, len(neg_results)))), columns = neg_df.columns)])

        AOPC_Plotter.aopc_plot(pos_df, neg_df, 'time (minutes)', 'AOPC - global', "optimization", opts, f'{model_type} {ds_name} aopc time')