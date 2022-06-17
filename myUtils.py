import numpy as np
import pickle
import pandas as pd

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
    
    
    def __init__(self, dataset, test, explainer, predict_fn, ignored, result_path = 'results/text_exps_bert.pickle', optimize = False):
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