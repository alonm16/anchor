import numpy as np

class MyExplenation:
    def __init__(self, index, fit_examples, test_cov, exp):
        self.index = index
        self.fit_examples = fit_examples
        self.test_cov = test_cov
        self.names =  exp.names()
        self.coverage = exp.coverage()
        self.precision = exp.precision()

class Utils:
    
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

    def remove_duplicates(self,explenations):
        exps_names = [' AND '.join(exp.names) for exp in explenations]
        seen = set()
        saved_exps = list()
        for i, exp in enumerate(explenations):
            if exps_names[i] not in seen:
                saved_exps.append(exp)
            seen.add(exps_names[i])

        return saved_exps

    def compute_explenations(self, indices):
        length = len(indices)
        explenations = list()
        for i, index in enumerate(indices):
            if i % 10 == 0:
                print(i)
            cur_exp = self.get_exp(index)
            cur_fit = self.get_fit_examples(cur_exp, index)
            cur_test_cov = self.get_test_cov(cur_fit)

            explenations.append(MyExplenation(index, cur_fit, cur_test_cov, cur_exp))

        return self.remove_duplicates(explenations)