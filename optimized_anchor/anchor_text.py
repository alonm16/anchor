from . import utils
from . import anchor_base
from . import anchor_explanation
import numpy as np
import json
import os
import string
import sys
from io import open
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from numba import jit, njit, float32
from numpy.random import default_rng

optimize = False

def id_generator(size=15):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))

# TODO optimization
@njit(cache=True)
def exp_normalize(x):  
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

class TextGenerator(object):
    def __init__(self, url=None, num_unmask = 500):
        self.url = url
        if url is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=False)
            if optimize:
                self.bert = torch.jit.load('models/mlm_models/distil_mlm.pt').to(self.device)
                self.ids_to_tokens = dict(self.bert_tokenizer.ids_to_tokens)
                self.num_unmask = num_unmask
            else:
                self.bert = DistilBertForMaskedLM.from_pretrained('distilbert-base-cased', torchscript=True)
                self.bert.to(self.device)
                self.bert.eval()
         
    # TODO: optimize, process multiple texts, get the texts already encoded
    def unmask(self, encoded_array):
        tokenizer = self.bert_tokenizer 
        model = self.bert
        ids_to_tokens = self.ids_to_tokens
        unk_token = tokenizer.unk_token
        encoded_array = np.array(encoded_array)
        masked_array = []
        #encoded_array = tokenizer(texts_with_mask, add_special_tokens=True, return_tensors="np", padding=True)["input_ids"]
        masked_array = [(encoded_array[i] == self.bert_tokenizer.mask_token_id).nonzero()[0] for i in range(len(encoded_array))]

        to_pred = torch.tensor(encoded_array, device=self.device)
        with torch.no_grad():
            outputs = model(to_pred)[0]
        rets = []  
        
        for j in range(len(encoded_array)):
            ret = []  

            if optimize:
                # optimize: change num of unmask options
                v_array, top_preds_array = torch.topk(outputs[j, masked_array[j]], self.num_unmask)
                top_preds_array = top_preds_array.tolist()
                v_array = v_array.cpu().numpy()

                for i in range(len(masked_array[j])):

                    v, top_preds = v_array[i], top_preds_array[i]
                    # optimize: doesn't convert the ids to tokens here, only later for the chosen tokens
                    ret.append((top_preds, v))

            else:
                for i in masked:
                    v, top_preds = torch.topk(outputs[0, i], 500)

                    words = tokenizer.convert_ids_to_tokens(top_preds)
                    v = np.array([float(x) for x in v])
                    
            rets.append(ret)
        
        return rets
    

class SentencePerturber:
    def __init__(self, words, tg, onepass=False):
        self.tg = tg
        self.words = words
        self.cache = {}
        self.mask = self.tg.bert_tokenizer.mask_token
        self.mask_id = self.tg.bert_tokenizer.mask_token_id
        self.array = np.array(words, '|U80')
        self.onepass = onepass
        self.rng = default_rng(42)
        self.choice = self.rng.choice
        self.tokenized_array = np.array([101] +[self.tg.bert_tokenizer.vocab[token] for token in self.words] + [102])
        self.pr = np.zeros(len(self.words))
        # optimize: changed beacuse of unmask optimization
        # not including [cls], [sep] tokens
        for i in range(len(words)):
            a = self.array.copy()
            a[i] = self.mask
            tokenized_a = self.tokenized_array.copy()
            tokenized_a[i+1] = self.mask_id
            s = ' '.join(a)
            # TODO: optimize, process multiple texts, so more [0]
            w, p = self.probs([s], [tokenized_a])[0][0]
            w = [self.tg.ids_to_tokens[w_i] for w_i in w]
            self.pr[i] =  min(0.5, dict(zip(w, p)).get(words[i], 0.01))
            
                
            
    # TODO: optimize, process multiple texts, and sending sentences as tokens, 
    # the sentence is encoded only once without masks, so only need to copy and change where the mask is
    def sample(self, data):
        arrays = []
        tokenized_arrays = []
        texts = []
        masks_array = []
        for i in range(data.shape[0]):
            a = self.array.copy()
            masks_array.append(np.where(data[i] == 0)[0])
            a[data[i] != 1] = self.mask
            texts.append(' '.join(a))
            arrays.append(a)
            tokenized_a = self.tokenized_array.copy()
            tokenized_a[1:-1][data[i] != 1] = self.mask_id
            tokenized_arrays.append(tokenized_a)
            
            
        if self.onepass:
            rs = self.probs(texts, tokenized_arrays)
            # TODO optimize : faster this way
            for i in range(len(rs)):
                if optimize:
                    reps = [self.tg.ids_to_tokens[a[self.choice(len(a), p=p)]] for a, p in rs[i]]
                    arrays[i][masks_array[i]] = reps
                else:
                    reps = [np.random.choice(a, p=p) for a, p in rs]
           
            #a[masks] = rep
        else:
            for i in masks:
                s = ' '.join(a)
                words, probs = self.probs(s)[0]
                a[i] = np.random.choice(words, p=probs)
        return arrays

    # TODO: optimize, process multiple texts
    def probs(self, texts, arrays):
        results = [None]*len(texts)
        not_cached =[]
        not_cached_idx = []
        for i, text in enumerate(texts):
            if text in self.cache:
                results[i] = self.cache[text]
            else:
                not_cached.append(arrays[i])
                not_cached_idx.append(i)
        if len(not_cached) > 0:
            rs = self.tg.unmask(not_cached)
            for i, r in zip(not_cached_idx, rs):
                results[i] = [(a, exp_normalize(b)) for a, b in r] 
                self.cache[texts[i]] = results[i] 
        
        # if s not in self.cache:
        #     r = self.tg.unmask(s)
        #     self.cache[s] = [(a, exp_normalize(b)) for a, b in r]
        #     if not self.onepass:
        #         self.cache[s] = self.cache[s][:1]
        return results


    def perturb_sentence(present, n, prob_change=0.5):
        raw = np.zeros((n, len(self.words)), '|U80')
        data = np.ones((n, len(self.words)))

#maybe use instead of np.random.choice
# @njit(int64(float32[:]), cache=True)
# def choice(p): 
#     return np.argmax(np.random.multinomial(1, p))
        
class AnchorText(object):
    """bla"""
    
    def __init__(self, nlp, class_names, use_unk_distribution=True, mask_string='UNK', num_unmask = 500):
        """
        Args:
            nlp: spacy object
            class_names: list of strings
            use_unk_distribution: if True, the perturbation distribution
                will just replace words randomly with mask_string.
                If False, words will be replaced by similar words using word
                embeddings
            mask_string: String used to mask tokens if use_unk_distribution is True.
        """
        self.nlp = nlp
        self.class_names = class_names
        self.use_unk_distribution = use_unk_distribution
        self.tg = None
        self.mask_string = mask_string
        if not self.use_unk_distribution:
            self.tg = TextGenerator(num_unmask = num_unmask)
            
    @staticmethod       
    def set_optimize(should_optimize):
        global optimize
        optimize = should_optimize 
        anchor_base.AnchorBaseBeam.set_optimize(should_optimize)
        

    def get_sample_fn(self, text, classifier_fn, onepass=False, use_proba=False):
        # for changed predict_sentences
        processed = self.tg.bert_tokenizer.tokenize(text)
        true_label = classifier_fn([processed])[0]
        words = np.array(processed, dtype='|U80')
        positions = [x.idx for x in self.nlp(text)]
        # positions = list(range(len(words)))
        perturber = None
        if not self.use_unk_distribution:
            perturber = SentencePerturber(words, self.tg, onepass=onepass)
        def sample_fn(present, num_samples, compute_labels=True):
            if self.use_unk_distribution:
                data = np.ones((num_samples, len(words)))
                raw = np.zeros((num_samples, len(words)), '|U80')
                raw[:] = words
                for i, t in enumerate(words):
                    if i in present:
                        continue
                    n_changed = np.random.binomial(num_samples, .5)
                    changed = np.random.choice(num_samples, n_changed,
                                               replace=False)
                    raw[changed, i] = self.mask_string
                    data[changed, i] = 0
                raw_data = [' '.join(x) for x in raw]
            else:
                data = np.zeros((num_samples, len(words)))
                for i in range(len(words)):
                    if i in present:
                        continue
                    
                    if optimize:
                        data[:, i] = np.random.binomial(1, perturber.pr[i], num_samples)
                    else:
                        probs = [1 - perturber.pr[i], perturber.pr[i]]
                        data[:, i] = np.random.choice([0, 1], num_samples, p=probs)
                data[:, present] = 1
                
                # TODO: optimize, process multiple texts
                raw_data = perturber.sample(data)
                data = raw_data[:] == words
         
            labels = []
            if compute_labels:
                with torch.no_grad():
                    labels = (classifier_fn(raw_data) == true_label).astype(int)
            labels = np.array(labels)
            return data, labels
        return words, positions, true_label, sample_fn
    

    def explain_instance(self, text, classifier_fn, ignored, threshold=0.95,
                          delta=0.1, tau=0.15, batch_size=10, onepass=False,
                          use_proba=False, beam_size=4, 
                          **kwargs):
        if type(text) == bytes:
            text = text.decode()
        words, positions, true_label, sample_fn = self.get_sample_fn(
            text, classifier_fn, onepass=onepass, use_proba=use_proba)
        # print words, true_label
        # TODO changed anchor_size
        anchor_base.AnchorBaseBeam.words = words
        anchor_base.AnchorBaseBeam.ignored = ignored
        
        exps = anchor_base.AnchorBaseBeam.anchor_beam(
            sample_fn, true_label, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, stop_on_first=True,
            coverage_samples=1, max_anchor_size=1, **kwargs)
        explanations = []
        for exp in exps:
            exp['names'] = [words[x] for x in exp['feature']]
            #exp['positions'] = [positions[x] for x in exp['feature']]
            exp['instance'] = text
            exp['prediction'] = true_label
            print(exp['precision'])
            explanation = anchor_explanation.AnchorExplanation('text', exp,
                                                               self.as_html)
            explanations.append(explanation)
            
        return explanations

    def as_html(self, exp):
        predict_proba = np.zeros(len(self.class_names))
        exp['prediction'] = int(exp['prediction'])
        predict_proba[exp['prediction']] = 1
        predict_proba = list(predict_proba)

        def jsonize(x):
            return json.dumps(x)
        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'), encoding='utf8').read()
        random_id = 'top_div' + id_generator()

        example_obj = []

        def process_examples(examples, idx):
            idxs = exp['feature'][:idx + 1]
            out_dict = {}
            new_names = {'covered_true': 'coveredTrue', 'covered_false': 'coveredFalse', 'covered': 'covered'}
            for name, new in new_names.items():
                ex = [x[0] for x in examples[name]]
                out = []
                for e in ex:
                    processed = self.nlp(str(e))
                    raw_indexes = [(processed[i].text, processed[i].idx, exp['prediction']) for i in idxs]
                    out.append({'text': e, 'rawIndexes': raw_indexes})
                out_dict[new] = out
            return out_dict

        example_obj = []
        for i, examples in enumerate(exp['examples']):
            example_obj.append(process_examples(examples, i))

        explanation = {'names': exp['names'],
                       'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                       'supports': exp['coverage'],
                       'allPrecision': exp['all_precision'],
                       'examples': example_obj}
        processed = self.nlp(exp['instance'])
        raw_indexes = [(processed[i].text, processed[i].idx, exp['prediction'])
                       for i in exp['feature']]
        raw_data = {'text': exp['instance'], 'rawIndexes': raw_indexes}
        jsonize(raw_indexes)

        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "text", "anchor");
        </script>'''.format(random_id=random_id,
                            label_names=jsonize(self.class_names),
                            predict_proba=jsonize(list(predict_proba)),
                            true_class=jsonize(False),
                            explanation=jsonize(explanation),
                            raw_data=jsonize(raw_data))
        out += u'</body></html>'
        return out

    def show_in_notebook(self, exp, true_class=False, predict_proba_fn=None):
        """Bla"""
        out = self.as_html(exp, true_class, predict_proba_fn)
        from IPython.core.display import display, HTML
        display(HTML(out))
