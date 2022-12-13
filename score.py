from collections import Counter
import pandas as pd
import numpy as np

class ScoreUtils:
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
    def calculate_agg_score(folder_name, tokenizer, anchor_examples, exps, labels, agg_name):
        aggs = {'sum': ScoreUtils.calculate_sum, 'avg': ScoreUtils.calculate_avg}
        columns = ['name', 'anchor score', 'type occurences', 'total occurences','+%', '-%', 'both', 'normal']

        pos_exps = [exp for exp in exps if labels[exp.index]==0]
        neg_exps = [exp for exp in exps if labels[exp.index]==1]

        anchor_occurences = ScoreUtils.get_anchor_occurences(exps)
        pos_occurences = ScoreUtils.get_anchor_occurences(pos_exps)
        neg_occurences = ScoreUtils.get_anchor_occurences(neg_exps)

        normal_occurences = ScoreUtils.get_normal_occurences(anchor_examples, anchor_occurences, tokenizer)
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
        
    @staticmethod
    def calculate_percent_scores(folder_name, tokenizer, anchor_examples, exps, labels, percent):
        """fix so not all sentences included!!!!!!!!!!!!!!"""
        """ calculates the scores for specific time during the running of anchor """
        alphas = [0.95, 0.8, 0.65, 0.5]
        dfs = []
        columns = ['name', 'anchor score', 'type occurences', 'total occurences','+%', '-%', 'both', 'normal']

        pos_exps = [exp for exp in exps if labels[exp.index]==0]
        neg_exps = [exp for exp in exps if labels[exp.index]==1]

        anchor_occurences = ScoreUtils.get_anchor_occurences(exps)
        pos_occurences = ScoreUtils.get_anchor_occurences(pos_exps)
        neg_occurences = ScoreUtils.get_anchor_occurences(neg_exps)

        normal_occurences = ScoreUtils.get_normal_occurences(anchor_examples, anchor_occurences, tokenizer)
        ScoreUtils.smooth_before(normal_occurences, [pos_occurences, neg_occurences])

        teta0 = ScoreUtils.calculate_teta0(normal_occurences)


        for alpha in alphas:
            df_pos, df_neg = [], []

            teta_pos = ScoreUtils.calculate_teta1(pos_occurences, teta0, alpha)
            ScoreUtils.smooth_after(teta_pos, pos_occurences)

            teta_neg = ScoreUtils.calculate_teta1(neg_occurences, teta0, alpha)
            ScoreUtils.smooth_after(teta_neg, neg_occurences)

            # substracting 1 because of the smoothing
            for anchor, score in teta_pos.items():
                pos_percent = round((pos_occurences[anchor]-1)/anchor_occurences[anchor], 2)
                neg_percent = 1-pos_percent
                both = (pos_occurences[anchor]-1)>0 and (neg_occurences[anchor]-1)>0
                df_pos.append([anchor, score , pos_occurences[anchor]-1, anchor_occurences[anchor], pos_percent, neg_percent, both,  normal_occurences[anchor]-1]) 


            for anchor, score in teta_neg.items():
                pos_percent = round((pos_occurences[anchor]-1)/anchor_occurences[anchor], 2)
                neg_percent = 1-pos_percent
                both = (pos_occurences[anchor]-1)>0 and (neg_occurences[anchor]-1)>0
                df_neg.append([anchor, score , neg_occurences[anchor]-1, anchor_occurences[anchor], pos_percent, neg_percent, both,  normal_occurences[anchor]-1]) 

            df_pos.sort(key=lambda exp: -exp[1])
            df_neg.sort(key=lambda exp: -exp[1])
            df_pos = pd.DataFrame(data = df_pos, columns = columns ).set_index('name')
            df_neg = pd.DataFrame(data = df_neg, columns = columns ).set_index('name')

            dfs.extend([df_pos, df_neg])

        writer = pd.ExcelWriter(f'{folder_name}/percents/scores-{percent}.xlsx', engine='xlsxwriter') 

        workbook=writer.book
        worksheet=workbook.add_worksheet('Sheet1')
        writer.sheets['Sheet1'] = worksheet

        cur_col = 0
        is_positive = False
        alphas = np.repeat(alphas, 2)

        for df, alpha in zip(dfs, alphas):
            cur_type = 'positive' if is_positive else 'negative'
            is_positive = not is_positive
            worksheet.write(0, cur_col, f'{alpha}-{cur_type}')
            df.to_excel(writer, sheet_name=f'Sheet1', startrow=1, startcol=cur_col)
            cur_col+= len(columns) + 1

        writer.save() 
        
    @staticmethod
    def calculate_scores(folder_name, tokenizer, anchor_examples, exps, labels):
        alphas = [0.95, 0.8, 0.65, 0.5]
        dfs = []
        columns = ['name', 'anchor score', 'type occurences', 'total occurences','+%', '-%', 'both', 'normal']
        
        pos_exps = [exp for exp in exps if labels[exp.index]==0]
        neg_exps = [exp for exp in exps if labels[exp.index]==1]

        anchor_occurences = ScoreUtils.get_anchor_occurences(exps)
        pos_occurences = ScoreUtils.get_anchor_occurences(pos_exps)
        neg_occurences = ScoreUtils.get_anchor_occurences(neg_exps)

        normal_occurences = ScoreUtils.get_normal_occurences(anchor_examples, anchor_occurences, tokenizer)
        ScoreUtils.smooth_before(normal_occurences, [pos_occurences, neg_occurences])

        teta0 = ScoreUtils.calculate_teta0(normal_occurences)


        for alpha in alphas:
            df_pos, df_neg = [], []

            teta_pos = ScoreUtils.calculate_teta1(pos_occurences, teta0, alpha)
            ScoreUtils.smooth_after(teta_pos, pos_occurences)

            teta_neg = ScoreUtils.calculate_teta1(neg_occurences, teta0, alpha)
            ScoreUtils.smooth_after(teta_neg, neg_occurences)

            # substracting 1 because of the smoothing
            for anchor, score in teta_pos.items():
                pos_percent = round((pos_occurences[anchor]-1)/anchor_occurences[anchor], 2)
                neg_percent = 1-pos_percent
                both = (pos_occurences[anchor]-1)>0 and (neg_occurences[anchor]-1)>0
                df_pos.append([anchor, score , pos_occurences[anchor]-1, anchor_occurences[anchor], pos_percent, neg_percent, both,  normal_occurences[anchor]-1]) 


            for anchor, score in teta_neg.items():
                pos_percent = round((pos_occurences[anchor]-1)/anchor_occurences[anchor], 2)
                neg_percent = 1-pos_percent
                both = (pos_occurences[anchor]-1)>0 and (neg_occurences[anchor]-1)>0
                df_neg.append([anchor, score , neg_occurences[anchor]-1, anchor_occurences[anchor], pos_percent, neg_percent, both,  normal_occurences[anchor]-1]) 

            df_pos.sort(key=lambda exp: -exp[1])
            df_neg.sort(key=lambda exp: -exp[1])
            df_pos = pd.DataFrame(data = df_pos, columns = columns ).set_index('name')
            df_neg = pd.DataFrame(data = df_neg, columns = columns ).set_index('name')

            dfs.extend([df_pos, df_neg])

        writer = pd.ExcelWriter(f'{folder_name}/scores.xlsx',engine='xlsxwriter') 

        workbook=writer.book
        worksheet=workbook.add_worksheet('Sheet1')
        writer.sheets['Sheet1'] = worksheet

        cur_col = 0
        is_positive = False
        alphas = np.repeat(alphas, 2)

        for df, alpha in zip(dfs, alphas):
            cur_type = 'positive' if is_positive else 'negative'
            is_positive = not is_positive
            worksheet.write(0, cur_col, f'{alpha}-{cur_type}')
            df.to_excel(writer, sheet_name=f'Sheet1', startrow=1, startcol=cur_col)
            cur_col+= len(columns) + 1

        writer.save()