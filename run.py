#!/usr/bin/env python
# coding: utf-8

import warnings
import spacy
from optimized_anchor import anchor_text, anchor_base
import pickle
import myUtils
from myUtils import *
from models.utils import *
from models.dataset_loader import *
import datetime
import time
import argparse
import os
import sys
sys.path.append('models')
# when apply torchscript to models sometimes
torch._C._jit_set_texpr_fuser_enabled(False)

parser = argparse.ArgumentParser()

SEED = 84
torch.manual_seed(SEED)
warnings.simplefilter("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sort_functions = {'polarity': sort_polarity, 'confidence': sort_confidence}

parser.add_argument("--dataset_name", default='sentiment', choices = ['sentiment', 'offensive', 'corona', 'toy-spam', "dilemma", "sport-spam", "home-spam"])
parser.add_argument("--model_type", default = 'tinybert', choices = ['tinybert', 'gru', 'svm', 'logistic'])
parser.add_argument("--sorting", default='confidence', choices=['polarity', 'confidence'])
parser.add_argument("--optimization", default='', choices = ['', 'topk', 'lossy', 'desired'], nargs = '+')
parser.add_argument("--examples_max_length", default=150, type=int)
parser.add_argument("--delta", default=0.1, type=float)

args = parser.parse_args()

examples_max_length = args.examples_max_length
do_ignore = 'lossy' in args.optimization
topk_optimize = 'topk' in args.optimization
desired_optimize = 'desired' in args.optimization
sort_function = sort_functions[args.sorting]

dataset_name = args.dataset_name
sorting = args.sorting
optimization = '-'.join(args.optimization)
optimization = '-'.join([optimization, str(args.delta)]) if args.optimization!='' else args.delta
model_type = args.model_type
model_name = 'huawei-noah/TinyBERT_General_4L_312D'
folder_name = f'results/{model_type}/{dataset_name}/{sorting}/{optimization}'

ds = get_ds(dataset_name)
model = load_model(f'models/{model_type}/{dataset_name}/traced.pt').to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
myUtils.model = model
myUtils.tokenizer = tokenizer

nlp = spacy.load('en_core_web_sm')

anchor_examples, true_labels = preprocess_examples(ds, examples_max_length)
anchor_examples, _ = sort_function(anchor_examples, true_labels)

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# In[6]:

normal_occurences = get_occurences(anchor_examples)
anchor_base.AnchorBaseBeam.best_group = BestGroup(folder_name, normal_occurences, filter_anchors = topk_optimize, desired_optimize = desired_optimize)

# In[7]:

if do_ignore:
    ignored = get_ignored(anchor_examples)
else:
    ignored = []


# In[10]:
print(folder_name)
print(datetime.datetime.now())

optimize = True
anchor_text.AnchorText.set_optimize(optimize)
explainer = anchor_text.AnchorText(nlp, ['positive', 'negative'], use_unk_distribution=False)


pickle.dump(anchor_examples, open( f"{folder_name}/anchor_examples.pickle", "wb" ))

st = time.time()

my_utils = TextUtils(anchor_examples, explainer, myUtils.predict_sentences, ignored, optimize = optimize, delta = args.delta)
set_seed()
#torch._C._jit_set_texpr_fuser_enabled(False)
explanations = my_utils.compute_explanations(list(range(len(anchor_examples))))

pickle.dump(explanations, open( f"{folder_name}/exps_list.pickle", "wb"))

print(datetime.datetime.now())

from csv import writer
with open('times.csv', 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow([folder_name[len('results/'):], (time.time()-st)/60, do_ignore, topk_optimize, desired_optimize ,examples_max_length])
