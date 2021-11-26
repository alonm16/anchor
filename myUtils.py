import numpy as np
import pickle

class MyExplanation:
    def __init__(self, index, fit_examples, test_cov, exp):
        self.index = index
        self.fit_examples = fit_examples
        self.test_cov = test_cov
        self.names =  exp.names()
        self.coverage = exp.coverage()
        self.precision = exp.precision()
        
class ExtendedExplanation:
    def __init__(self, exp, anchor_examples, test, test_labels, predict_sentences, explainer):
        self.index = exp.index
        self.fit_examples = exp.fit_examples
        self.test_cov = exp.test_cov
        self.names =  exp.names
        self.coverage = exp.coverage
        self.precision = exp.precision
        exp_label =  predict_sentences([str(anchor_examples[exp.index])])[0]
        self.test_precision = np.mean(predict_sentences(test[exp.fit_examples]) == exp_label)
        self.real_precision = np.mean(test_labels[exp.fit_examples] == explainer.class_names[exp_label])

class TabularUtils:
    
    def __init__(self, dataset, explainer, c):
        self.dataset = dataset
        self.explainer = explainer
        self.c = c

    def get_exp(self,idx):
        return self.explainer.explain_instance(self.dataset.test[idx], self.c.predict, threshold=0.95)

    def get_fit_examples(self,exp, idx):
        return np.where(np.all(self.dataset.test[:, exp.features()] == self.dataset.test[idx][exp.features()], axis=1))[0]

    def get_test_cov(self,fit_anchor):
        return (fit_anchor.shape[0] / float(self.dataset.test.shape[0]))

    def remove_duplicates(self,explanations):
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
        for i, index in enumerate(indices):
            if i % 50 == 0:
                print(i)
            cur_exp = self.get_exp(index)
            cur_fit = self.get_fit_examples(cur_exp, index)
            cur_test_cov = self.get_test_cov(cur_fit)

            explanations.append(MyExplanation(index, cur_fit, cur_test_cov, cur_exp))
        
        explanations = self.remove_duplicates(explanations)
        explanations.sort(key=lambda exp: exp.test_cov)
        
        return explanations
    
    
class TextUtils:
    
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

    def remove_duplicates(self,explanations):
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