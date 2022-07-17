import numpy as np
import pickle
import pandas as pd
import torch
import spacy
import numpy as np
import random

model = None
text_parser = None
nlp = spacy.load('en_core_web_sm')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1 = pad 2=sos 3 = eos
def tokenize(text, max_len):
    sentence = nlp.tokenizer(str(text))
    input_tokens = [2] + [text_parser.vocab.stoi[word.text] for word in sentence] + [3] + [1]*(max_len-len(sentence))

    return input_tokens

def predict_sentences(sentences):
    half_length = len(sentences)//2
    if(half_length>100):
        return np.concatenate([predict_sentences(sentences[:half_length]), predict_sentences(sentences[half_length:])])
    max_len = max([len(sentence) for sentence in sentences])
    sentences = torch.tensor([tokenize(sentence, max_len) for sentence in sentences], device=device)
    input_tokens = torch.transpose(sentences, 0, 1)
    output = model(input_tokens)

    return torch.argmax(output, dim=1).cpu().numpy()

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
            c.update(text_parser.tokenize(sentence))
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

def sort_sentences(sentences, dataset_name):
    """score calculated as average absolute positivity/negativity of non stop words, normalized by their non stop words percentage"""
    stop_words = get_stopwords()
    tok_sentences = [text_parser.tokenize(sentence) for sentence in sentences]
    predictions = [predict_sentences([str(anchor_example)])[0] for anchor_example in sentences]
    words_distribution = get_words_distribution(tok_sentences, predictions, stop_words)
    
    def sentence_score(sentence):
        num_stops = sum(word in stop_words for word in sentence)
        avg_occurences = sum(words_distribution[word] for word in sentence if word not in stop_words)/(len(sentence)-num_stops)
        
        return avg_occurences*(len(sentence) - num_stops)/len(sentence)
    
    scored_sentences = [(sentences[i], sentence_score(sentence)) for i, sentence in enumerate(tok_sentences)]
    scored_sentences.sort(key=lambda exp: -abs(exp[1]))
    return [exp[0] for exp in scored_sentences]

class BestGroup:
    def __init__(self, occurences):
        self.occurences_left = occurences
        self.best = defaultdict(int)
        self.all = defaultdict(int)
        self.normal = defaultdict(int)
        self.min_val = 0
        self.min_name = None
        self.full = False
        self.normal_factor = 0.2
    
    def update(self, anchor):
        self.all[anchor]+=1
        
        if anchor in self.best:
            self.best[anchor]+=1
            if anchor == self.min_name:
                self._update_min(anchor, self.best[anchor])
        elif not self.full:
            self.best[anchor] = self.all[anchor]
            
            if len(self.best)==50:
                self.full = True
                self._update_min(anchor, self.best[anchor])
        # in case anchor with equal value was outside the best
        elif self.all[anchor] > self.min_val:
            del self.best[self.min_name]
            self.best[anchor] = self.all[anchor]
            self._update_min(anchor, self.best[anchor]) 
            
    def _update_min(self, candid_name, candid_val):
        for anchor, value in self.best.items():
            if value < candid_val:
                candid_name, candid_val = anchor, value
                break
                
        self.min_name = candid_name
        self.min_val = candid_val
    
    def should_calculate(self, anchor):
        should = (self.all[anchor]+self.occurences_left[anchor] - self.normal_factor*self.normal[anchor]) >= (self.min_val - self.normal_factor*self.normal[self.min_name])
        self.occurences_left[anchor]-=1

        return should
        

    

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
    def __init__(self, dataset, test, explainer, predict_fn, ignored, result_path, optimize = False):
        self.dataset = dataset
        self.explainer = explainer
        self.predict_fn = predict_fn
        self.test = test
        self.path = result_path
        self.ignored = ignored
        
        explainer.set_optimize(optimize)

    def get_exp(self,idx):
        return self.explainer.explain_instance(self.dataset[idx], self.predict_fn, self.ignored, threshold=0.95, verbose=False, onepass=True)

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