from collections import Counter, defaultdict
import pandas as pd
import numpy as np

class ScoreUtils:
    columns=['name','anchor score','type occurences','total occurences','+%','-%','both', 'normal']
        
    @staticmethod
    def get_scores_dict(folder_name, trail_path = "0.1/scores.xlsx", alpha = 0.95):
        """
        returns dict of (anchor, score) pairs, and sum of the topk positive/negative
        """
        df = pd.read_excel(f'{folder_name}{trail_path}').drop(0)
        index_prefix = f"{alpha}-" if alpha is not None else ""
        
        def get_scores(column):
            # filter_idx = list(df.columns).index(column)+2
            # filtered_df = df[df.iloc[:, filter_idx]>5]
            # keys = filtered_df[column].dropna().tolist()
            #values = filtered_df.iloc[:, list(df.columns).index(column)+1].tolist()
            keys = df[column].dropna().tolist()
            values = df.iloc[:, list(df.columns).index(column)+1].tolist()
            return dict(zip(keys, values))  
        
        return get_scores(f'{index_prefix}positive'), get_scores(f'{index_prefix}negative')
    
    @staticmethod
    def get_normal_occurences(sentences, anchor_occurences, tokenizer):
        c = Counter()
        for sentence in sentences:
            c.update(tokenizer.tokenize(sentence))
            
        c.subtract(anchor_occurences)
        return c
    
    @staticmethod
    def get_occurences(sentences, exps, labels, tokenizer):
        anchor_occurences = Counter(map(lambda e: e.names[0], exps))
        pos_occurences = Counter([e.names[0] for e in exps if labels[e.index]==1])
        neg_occurences = Counter([e.names[0] for e in exps if labels[e.index]==0])
        normal_occurences = ScoreUtils.get_normal_occurences(sentences, anchor_occurences, tokenizer)
        return anchor_occurences, pos_occurences, neg_occurences, normal_occurences

    @staticmethod
    def calculate_sum(anchor_occurences, normal_occurences, min_occurrences=0, ds_size=3400):
        sums = dict()
        sum_occurences = sum(anchor_occurences.values())
        for word, count in anchor_occurences.items():
            sums[word] = count/sum_occurences

        return sums

    @staticmethod
    def calculate_avg(anchor_occurences, normal_occurences, min_percent=0, ds_size=3400):
        avgs = dict()
        min_count = min_percent*ds_size
        for word, count in anchor_occurences.items():
            occurrences = anchor_occurences[word]+normal_occurences[word]
            if anchor_occurences[word] > min_count:
                avgs[word] = count/occurrences

        return avgs
    
    @staticmethod
    def smooth_before(normal_occurences, anchor_occurences_list):
        for w in normal_occurences:
            normal_occurences[w]+=1
            for anchor_occurences in anchor_occurences_list:
                anchor_occurences[w]+=1

    @staticmethod
    def smooth_after(teta1, type_occurences):
        # removing words we added 1 at the start smooth
        words = list(teta1.keys())
        for word in words:
            if type_occurences[word]<=1:
                del teta1[word]

        min_val = min(teta1.values(), default = 0) 
        if min_val<0:
            for w in teta1:
                teta1[w]-= min_val
            sum_val = sum(teta1.values())
            for w in teta1:
                teta1[w]= teta1[w]/sum_val
    
    @staticmethod
    def calculate_teta0(normal_occurences):
        teta0 = dict()
        sum_occurences = sum(normal_occurences.values())
        for word, count in normal_occurences.items():
            teta0[word] = count/sum_occurences

        return teta0

    @staticmethod
    def calculate_teta1(anchor_occurences, teta0, alpha):
        teta1 = dict()
        sum_occurences = sum(anchor_occurences.values())
        for word, count in anchor_occurences.items():
            teta1[word] = count/sum_occurences -(1-alpha)*teta0[word]
            teta1[word] = teta1[word]/alpha

        return teta1
    
    @staticmethod
    def score_df(type_occurences, pos_occurences, neg_occurences, anchor_occurences, normal_occurences, teta0, alpha):
        df = []
        teta = ScoreUtils.calculate_teta1(type_occurences, teta0, alpha)
        ScoreUtils.smooth_after(teta, type_occurences)
        
        for anchor, score in teta.items():
            # substracting 1 because of the smoothing
            pos_percent = round((pos_occurences[anchor]-1)/anchor_occurences[anchor], 2)
            neg_percent = 1-pos_percent
            both = (pos_occurences[anchor]-1)>0 and (neg_occurences[anchor]-1)>0
            df.append([anchor, score , type_occurences[anchor]-1, anchor_occurences[anchor], pos_percent, neg_percent, both, normal_occurences[anchor]-1]) 

        df.sort(key=lambda exp: -exp[1])
        return pd.DataFrame(data = df, columns = ScoreUtils.columns).set_index('name')
    
    @staticmethod
    def calculate_agg_score(folder_name, tokenizer, sentences, exps, labels, agg_name, min_count=0):
        aggs = {'sum': ScoreUtils.calculate_sum, 'avg': ScoreUtils.calculate_avg}
        alphas = [0.95, 0.8, 0.65, 0.5]
        anchor_occurences, pos_occurences, neg_occurences, normal_occurences = ScoreUtils.get_occurences(sentences, exps, labels, tokenizer)
        df_pos, df_neg = [], []

        teta_pos = aggs[agg_name](pos_occurences, normal_occurences, min_count, len(sentences))
        teta_neg = aggs[agg_name](neg_occurences, normal_occurences, min_count, len(sentences))

        for anchor, score in teta_pos.items():
            pos_percent = round((pos_occurences[anchor])/anchor_occurences[anchor], 2)
            neg_percent = 1-pos_percent
            both = pos_occurences[anchor]>0 and neg_occurences[anchor]>0
            df_pos.append([anchor, score , pos_occurences[anchor], anchor_occurences[anchor], pos_percent, neg_percent, both, normal_occurences[anchor]]) 

        for anchor, score in teta_neg.items():
            pos_percent = round((pos_occurences[anchor])/anchor_occurences[anchor], 2)
            neg_percent = 1-pos_percent
            both = pos_occurences[anchor]>0 and neg_occurences[anchor]>0
            df_neg.append([anchor, score, neg_occurences[anchor], anchor_occurences[anchor], pos_percent, neg_percent, both, normal_occurences[anchor]])

        df_pos.sort(key=lambda exp: -exp[1])
        df_neg.sort(key=lambda exp: -exp[1])
        df_pos = pd.DataFrame(data = df_pos, columns = ScoreUtils.columns).set_index('name')
        df_neg = pd.DataFrame(data = df_neg, columns = ScoreUtils.columns).set_index('name')

        agg_name = agg_name if min_count == 0 else agg_name+f'_{min_count}'
        with pd.ExcelWriter(f'{folder_name}/{agg_name}_scores.xlsx', engine='xlsxwriter') as writer:
            cur_type = 'positive'
            cur_col = 0
            
            for df in [df_neg, df_pos]:
                cur_type = 'positive' if cur_type=='negative' else 'negative'
                df.to_excel(writer, sheet_name=f'Sheet1', startrow=1, startcol=cur_col)
                writer.book.worksheets()[0].write(0, cur_col, f'{cur_type}')
                cur_col+= len(ScoreUtils.columns) + 1
        
    @staticmethod
    def calculate_percent_scores(folder_name, tokenizer, sentences, exps, labels, percent):
        """ calculates the scores for specific time during the running of anchor """
        pos_sentences = [s for s, l in zip(sentences, labels) if l==1]
        pos_indices = [i for i, l in enumerate(labels) if l==1]
        pos_index = int((percent*len(pos_sentences)/100))
        pos_exps = list(filter(lambda e: labels[e.index]==1 and pos_indices.index(e.index)<=pos_index, exps))
 
        neg_sentences = [s for s, l in zip(sentences, labels) if l==0]
        neg_indices = [i for i, l in enumerate(labels) if l==0]
        neg_index = int((percent*len(neg_sentences)/100))
        neg_exps = list(filter(lambda e: labels[e.index]==0 and neg_indices.index(e.index)<=neg_index, exps))
        sentences = pos_sentences[:pos_index] + neg_sentences[:neg_index]
    
        path = f'{folder_name}/percents/scores-{percent}.xlsx'
        ScoreUtils.calculate_scores(path, tokenizer, sentences, pos_exps + neg_exps, labels)

    @staticmethod
    def calculate_scores(path, tokenizer, sentences, exps, labels):
        alphas = [0.95, 0.8, 0.65, 0.5]
        dfs = []
        anchor_occurences, pos_occurences, neg_occurences, normal_occurences = ScoreUtils.get_occurences(sentences, exps, labels, tokenizer)

        ScoreUtils.smooth_before(normal_occurences, [pos_occurences, neg_occurences])
        teta0 = ScoreUtils.calculate_teta0(normal_occurences)

        for alpha in alphas:
            df_pos = ScoreUtils.score_df(pos_occurences, pos_occurences, neg_occurences, anchor_occurences, normal_occurences, teta0, alpha)
            df_neg = ScoreUtils.score_df(neg_occurences, pos_occurences, neg_occurences, anchor_occurences, normal_occurences, teta0, alpha) 
            dfs.extend([df_neg, df_pos])
        
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            cur_type = 'positive'
            cur_col = 0

            for df, alpha in zip(dfs, np.repeat(alphas, 2)):
                cur_type = 'positive' if cur_type=='negative' else 'negative'  
                df.to_excel(writer, startrow=1, startcol=cur_col)
                writer.book.worksheets()[0].write(0, cur_col, f'{alpha}-{cur_type}')
                cur_col+= len(ScoreUtils.columns) + 1

    @staticmethod
    def calculate_time_scores(tokenizer, sentences, exps, labels, alphas = [0.95, 0.8, 0.65, 0.5]):
        """ calculates the scores for specific time during the running of anchor """
        pos_sentences = [s for s, l in zip(sentences, labels) if l==1]
        pos_indices = [i for i, l in enumerate(labels) if l==1]
        neg_sentences = [s for s, l in zip(sentences, labels) if l==0]
        neg_indices = [i for i, l in enumerate(labels) if l==0]
        
        results = defaultdict(lambda: defaultdict(dict))
        for percent in range(5, 105, 5):
            pos_index = int(percent*len(pos_sentences)/100)
            pos_exps = list(filter(lambda e: labels[e.index]==1 and pos_indices.index(e.index)<=pos_index, exps))

            neg_index = int(percent*len(neg_sentences)/100)
            neg_exps = list(filter(lambda e: labels[e.index]==0 and neg_indices.index(e.index)<=neg_index, exps))

            sentences = pos_sentences[:pos_index] + neg_sentences[:neg_index]
            
            anchor_occurences, pos_occurences, neg_occurences, normal_occurences = ScoreUtils.get_occurences(sentences, pos_exps+neg_exps, labels, tokenizer)

            ScoreUtils.smooth_before(normal_occurences, [pos_occurences, neg_occurences])
            teta0 = ScoreUtils.calculate_teta0(normal_occurences)    
            
            for alpha in alphas:
                results[alpha]['pos'][percent] = ScoreUtils.score_df(pos_occurences, pos_occurences, neg_occurences, anchor_occurences, normal_occurences, teta0, alpha)[[ScoreUtils.columns[1]]]
                results[alpha]['neg'][percent] = ScoreUtils.score_df(neg_occurences, pos_occurences, neg_occurences, anchor_occurences, normal_occurences, teta0, alpha)[[ScoreUtils.columns[1]]]
            
        return results