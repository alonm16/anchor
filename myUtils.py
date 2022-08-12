import numpy as np
import pickle
import pandas as pd
import torch
import spacy
import numpy as np
import random
from csv import writer
import time

model = None
tokenizer = None
nlp = spacy.load('en_core_web_sm')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def predict_sentences(sentences):
    encoded = [[101] +[tokenizer.vocab[token] for token in tokens] + [102]         
               for tokens in sentences]
    #encoded = tokenizer.encode(sentences, add_special_tokens=True, return_tensors="pt").to(device)
    to_pred = torch.tensor(encoded, device=device)
    outputs = model(to_pred)[0]
    return torch.argmax(outputs, dim=1).cpu().numpy()


from nltk.corpus import stopwords
def get_stopwords():
    stop_words = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.', 
                  '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', 
                  '`', '{', '|', '}', '~', '»', '«', '“', '”', '\s', '\t']
    stop_words.extend(["--", "'s", 'sos', 'eos'])
    stop_words.extend(stopwords.words('english'))
    return stop_words

from collections import Counter, defaultdict
def get_ignored(anchor_sentences):
    stop_words = get_stopwords()
    
    def get_below_occurences(sentences):
        min_value = 1
        c = Counter()
        for sentence in sentences:
            c.update(nlp.tokenizer(sentence))
        return set(w for w in c if c[w]<=min_value)

    return set(stop_words).union(get_below_occurences(anchor_sentences))

def get_occurences(sentences):
    c = Counter()
    for sentence in sentences:
        c.update([x.text for x in nlp.tokenizer(sentence)])
        
    return c

def get_words_distribution(sentences, predictions, stop_words):
    """ returns score between [-1, 1] of how much the word is in positive predicted sentences, ignoring stop words"""
    c_pos = Counter()
    c_neg = Counter()
    c = Counter()
    
    for sentence, prediction in zip(sentences, predictions):
        if prediction == 1:
            
            c_pos.update(set(sentence))
        else:
            c_neg.update(set(sentence))

    all_words = list(c_pos.keys())
    all_words.extend(c_neg.keys())
    all_words = set(all_words)
    
    for word in all_words:
        c[word] = (c_pos[word]-c_neg[word])/(c_pos[word]+c_neg[word])
    for word in stop_words:
        c[word]=0
    
    return c


def sort_polarity(sentences):
    """score calculated as average absolute positivity/negativity of non stop words, normalized by their non stop words percentage"""
    stop_words = get_stopwords()
    tok_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    predictions = [predict_sentences([example])[0] for example in tok_sentences]
    words_distribution = get_words_distribution(tok_sentences, predictions, stop_words)
    
    def sentence_score(sentence):
        num_stops = sum(word in stop_words for word in sentence)
        avg_occurences = sum(words_distribution[word] for word in sentence if word not in stop_words)/(len(sentence)-num_stops)
        
        return avg_occurences*(len(sentence) - num_stops)/len(sentence)
    
    scored_sentences = [(sentences[i], sentence_score(sentence)) for i, sentence in enumerate(tok_sentences)]
    scored_sentences.sort(key=lambda exp: -abs(exp[1]))
    return [exp[0] for exp in scored_sentences]

def sort_confidence(sentences):
    """sorts them according to the prediction confidence"""
    softmax = torch.nn.Softmax()
    
    def predict_sentence_logits(sentence):
        encoded = tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt").to(device)
        return softmax(model(encoded)[0])
    
    predictions = [predict_sentence_logits(sentence)[0] for sentence in sentences]
    predictions_confidence = [prediction[torch.argmax(prediction, dim=-1).item()] for prediction in predictions]
    
    scored_sentences = [(sentence, predictions_confidence[i]) for i, sentence in enumerate(sentences)]
    scored_sentences.sort(key=lambda exp: -abs(exp[1]))
    return [exp[0] for exp in scored_sentences]
    

class BestGroupInner:
    # better to update when a word is found normal, 
    # and in keep all anchors sorted in case someone out of the best became 
    # better than min because normal decreased the value of the min
    def __init__(self, occurences, normal_counts, group_size=25):
        self.occurences_left = occurences
        self.best = set()
        self.anchor_counts = defaultdict(int)
        self.normal_counts = normal_counts
        self.min_name = None
        self.full = False
        self.normal_factor = 0.1
        self.group_size = group_size
    
    def update(self, anchor):
        self.anchor_counts[anchor]+=1
        
        if anchor == self.min_name:
            self._update_min(anchor)
        elif not self.full:
            self.best.add(anchor)
            
            if len(self.best)==self.group_size:
                self._update_min(anchor)
                self.full = True
                
        # in case anchor with equal value was outside the best
        elif (anchor not in self.best) and self.pseudo_score(anchor) > self.pseudo_score(self.min_name):
            self.best.remove(self.min_name)
            self.best.add(anchor)
            self._update_min(anchor) 
            
    def pseudo_score(self, anchor):
        return self.anchor_counts[anchor] - self.normal_factor*self.normal_counts[anchor]
            
    def _update_min(self, candid_name):
        candid_val = self.pseudo_score(candid_name)
        
        for anchor in self.best:
            cur_score = self.pseudo_score(anchor)
            if cur_score < candid_val:
                candid_name = anchor
                candid_val = cur_score
                
        self.min_name = candid_name
    
    def should_calculate(self, anchor):
        should = (self.pseudo_score(anchor) + self.occurences_left[anchor]) >= self.pseudo_score(self.min_name)
        self.occurences_left[anchor]-=1

        return should
    
class BestGroup:
    def __init__(self, folder_name, occurences, filter_anchors = False, desired_optimize = False ,group_size = 50):
        self.occurences_left = occurences
        self.normal_counts = defaultdict(int)
        self.pos_BG = BestGroupInner(occurences, self.normal_counts, group_size//2)
        self.neg_BG = BestGroupInner(occurences, self.normal_counts, group_size//2)
        self.cur_type = None
        self.pos_monitor_writer = writer(open(f'{folder_name}/pos_monitor.csv', 'a+', newline=''))
        self.neg_monitor_writer = writer(open(f'{folder_name}/neg_monitor.csv', 'a+', newline=''))
        self.time_monitor_writer = writer(open(f'{folder_name}/time_monitor.csv', 'a+', newline=''))
        self.filter_anchors = filter_anchors
        self.desired_optimize = desired_optimize
        self.st = time.time()
        
    def update_anchor(self, anchor):
        if self.cur_type == 1:
            self.pos_BG.update(anchor)
        else:
            self.neg_BG.update(anchor)
    
    def update_normal(self, anchor):
        self.normal_counts[anchor]+=1
        
    def should_calculate(self, anchor):
        if not self.filter_anchors:
            return True
        if self.cur_type == 1:
            return self.pos_BG.should_calculate(anchor)
        else:
            return self.neg_BG.should_calculate(anchor)
    
    def desired_confidence_factor(self, anchor):
        """ 
        NOT RELATED TO TOPK OPTIMIZATION
        substract this factor from the desired confidence so if a word occured a lot as anchor, we need to calculate less
        formula: 0.2*(#anchor - normal_factor * #normal)/#all
        thus as the algorithm progresses the factor gets higher for the same pseudo score
        multiply by 0.2 so the factor is in range (0, 0.2)
        """
        if not self.desired_optimize:
            return 0
        all_occurences = self.occurences_left[anchor] + self.normal_counts[anchor] + self.pos_BG.anchor_counts[anchor] + self.neg_BG.anchor_counts[anchor]
        pseudo_score = 0
        if self.cur_type == 1:
            pseudo_score = self.pos_BG.pseudo_score(anchor)
        else:
            pseudo_score = self.neg_BG.pseudo_score(anchor)
        
        return 0.4*pseudo_score/all_occurences
        
    def monitor(self):
        self.pos_monitor_writer.writerow(list(self.pos_BG.best))
        self.neg_monitor_writer.writerow(list(self.neg_BG.best))
        self.time_monitor_writer.writerow([(time.time()-self.st)/60])

    

class MyExplanation:
    def __init__(self, index, fit_examples, test_cov, exp):
        self.index = index
        self.fit_examples = fit_examples
        self.test_cov = test_cov
        self.names = exp.names()
        self.coverage = exp.coverage()
        self.precision = exp.precision()
        
class ExtendedExplanation:
    def __init__(self, exp, anchor_examples, test, test_labels, test_predictions, predict_sentences, explainer):
        self.index = exp.index
        self.fit_examples = exp.fit_examples
        self.test_cov = exp.test_cov
        self.names =  exp.names
        self.coverage = exp.coverage
        self.precision = exp.precision
        exp_label =  predict_sentences([str(anchor_examples[exp.index])])[0]
        self.test_precision = np.mean(test_predictions[exp.fit_examples] == exp_label)
        #prediction is opposite depending on db
        self.real_precision = 1-np.mean(test_labels[exp.fit_examples] == exp_label)
        #self.real_precision = np.mean(test_labels[exp.fit_examples] == exp_label)
    
# utils for the MODIFIED anchor algorithm
class TextUtils:
    def __init__(self, dataset, test, explainer, predict_fn, ignored, result_path, optimize = False, delta=0.1):
        self.dataset = dataset
        self.explainer = explainer
        self.predict_fn = predict_fn
        self.test = test
        self.path = result_path
        self.ignored = ignored
        self.delta = delta
        
        explainer.set_optimize(optimize)

    def get_exp(self,idx):
        return self.explainer.explain_instance(self.dataset[idx], self.predict_fn, self.ignored, delta=self.delta, threshold=0.95, verbose=False, onepass=True)

    def get_fit_examples(self,exp):
        #irrelevant
        return []
        exp_words = exp.names()
        is_holding = [all(word in example for word in exp_words) for example in self.test]
        return np.where(is_holding)[0]

    def get_test_cov(self,fit_anchor):
        return (len(fit_anchor) / float(len(self.test)))

    @staticmethod
    def remove_duplicates(explanations):
        exps_names = [' AND '.join(exp.names) for exp in explanations]
        seen = set()
        saved_exps = list()
        for i, exp in enumerate(explanations):
            if exps_names[i] not in seen:
                saved_exps.append(exp)
                seen.add(exps_names[i])

        return saved_exps

    def compute_explanations(self, indices):
        explanations = list()
        with open(self.path, 'wb') as fp:
            for i, index in enumerate(indices):

                print('number '+str(i))
                cur_exps = self.get_exp(index)
                for cur_exp in cur_exps:
                    cur_fit = self.get_fit_examples(cur_exp)
                    cur_test_cov = self.get_test_cov(cur_fit)

                    explanation = MyExplanation(index, cur_fit, cur_test_cov, cur_exp)
                    explanations.append(explanation)
                    pickle.dump(explanation, fp)
                    fp.flush()


        #explanations = self.remove_duplicates(explanations)
        explanations.sort(key=lambda exp: exp.test_cov)
        
        return explanations
    
# utils for the ORIGINAL anchor algorithm
class OrigTextUtils:
    
    def __init__(self, dataset, test, explainer, predict_fn, result_path = 'results/text_exps_bert.pickle'):
        self.dataset = dataset
        self.explainer = explainer
        self.predict_fn = predict_fn
        self.test = test
        self.path = result_path

    def get_exp(self,idx):
        return self.explainer.explain_instance(self.dataset[idx], self.predict_fn, threshold=0.95, verbose=False, onepass=True)

    def get_fit_examples(self,exp):
        exp_words = exp.names()
        is_holding = [all(word in example for word in exp_words) for example in self.test]
        return np.where(is_holding)[0]

    def get_test_cov(self,fit_anchor):
        return (len(fit_anchor) / float(len(self.test)))

    @staticmethod
    def remove_duplicates(explanations):
        exps_names = [' AND '.join(exp.names) for exp in explanations]
        seen = set()
        saved_exps = list()
        for i, exp in enumerate(explanations):
            if exps_names[i] not in seen:
                saved_exps.append(exp)
                seen.add(exps_names[i])

        return saved_exps

    def compute_explanations(self, indices):
        explanations = list()
        with open(self.path, 'wb') as fp:
            for i, index in enumerate(indices):

                print('number '+str(i))
                cur_exp = self.get_exp(index)
                cur_fit = self.get_fit_examples(cur_exp)
                cur_test_cov = self.get_test_cov(cur_fit)

                explanation = MyExplanation(index, cur_fit, cur_test_cov, cur_exp)
                explanations.append(explanation)
                pickle.dump(explanation, fp)
                fp.flush()


        explanations = self.remove_duplicates(explanations)
        explanations.sort(key=lambda exp: exp.test_cov)
        
        return explanations
