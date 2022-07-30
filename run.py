#!/usr/bin/env python
# coding: utf-8

import warnings
import spacy
from optimized_anchor import anchor_text, anchor_base
import pickle
import myUtils
from myUtils import *
from transformer.utils import *
from dataset.dataset_loader import *
import datetime
import time
import argparse
parser = argparse.ArgumentParser()

SEED = 84
torch.manual_seed(SEED)
warnings.simplefilter("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sort_functions = {'polarity': sort_sentences, 'confidence': sort_sentences_confidence}

parser.add_argument("--dataset_name", default='sentiment', choices = ['sentiment', 'offensive', 'corona', 'sentiment_twitter'])
parser.add_argument("--optimization", default='')
parser.add_argument("--sort_function", default='polarity', choices=['polarity', 'confidence'])
parser.add_argument("--examples_max_length", default=90, type=int)
args = parser.parse_args()

examples_max_length = args.examples_max_length
do_ignore = args.optimization=='lossy'
topk_optimize = args.optimization.startswith('topk')
sort_function = sort_functions[args.sort_function]

# can be sentiment/spam/offensive/corona
dataset_name = args.dataset_name
optimization = args.optimization
folder_name = f'results/{dataset_name}-{optimization}' if len(optimization)>0 else f'results/{dataset_name}'
text_parser, label_parser, ds_train, ds_val = get_dataset(dataset_name)

# In[3]:

model = load_model('gru' , f'transformer/{dataset_name}/gru.pt', text_parser)
myUtils.model = torch.jit.script(model)
myUtils.text_parser = text_parser


# In[4]:

nlp = spacy.load('en_core_web_sm')


# In[5]:


train, train_labels, test, test_labels, anchor_examples = preprocess_examples(ds_train, examples_max_length)

anchor_examples = sort_function(anchor_examples)


# In[6]:

normal_occurences = get_occurences(anchor_examples)
anchor_base.AnchorBaseBeam.best_group = BestGroup(folder_name, normal_occurences, filter_anchors = topk_optimize)

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


# In[16]:


pickle.dump( test, open(f"{folder_name}/test.pickle", "wb" ))
pickle.dump( test_labels, open( f"{folder_name}/test_labels.pickle", "wb" ))
pickle.dump( anchor_examples, open( f"{folder_name}/anchor_examples.pickle", "wb" ))

st = time.time()
    
my_utils = TextUtils(anchor_examples, test, explainer, predict_sentences_optimized, ignored, f"profile.pickle", optimize = True)
set_seed()
explanations = my_utils.compute_explanations(list(range(len(anchor_examples))))

pickle.dump( explanations, open( f"{folder_name}/profile_list.pickle", "wb"))


# In[ ]:


print(datetime.datetime.now())

from csv import writer
with open('times.csv', 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow([folder_name[len('results/'):], (time.time()-st)/60, do_ignore, topk_optimize, examples_max_length])
