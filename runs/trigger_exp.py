#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Setup
import sys
sys.path.append('..')
import time
import matplotlib.pyplot as plt
import warnings
import torch.nn as nn
import spacy
from anchor import anchor_text
import pickle
import torch
from myUtils import *
from triggers.model_loader import load_model
from triggers.data_utils import *

SEED = 84
torch.manual_seed(SEED)
warnings.simplefilter("ignore")


# In[2]:


plt.rcParams['font.size'] = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[3]:


model, vocab = load_model()


# In[4]:


dev, dev_labels = get_dev_sst()


# In[5]:


from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer as Tokenizer
from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from allennlp.nn.util import move_to_device

t = Tokenizer()


# In[6]:


def predict_sentences(sentences):
    sentences = [create_instance(s,t) for s in sentences]
    iterator = SimpleDataLoader(sentences, batch_size = len(sentences))
    iterator.index_with(vocab)
    
    input_tokens = list(iterator)[0]['tokens']
    input_tokens = move_to_device(input_tokens, device=0)
    
    output = model(input_tokens)
    return torch.argmax(output, dim=1).cpu().numpy()


# In[7]:


predict_sentences(["good"])


# In[8]:


len([dev_x for dev_x in dev if 80>len(dev_x) > 15])


# # Prediction is opposite to label!!

# # Anchor Part

# In[9]:


nlp = spacy.load('en_core_web_sm')


# In[10]:


explainer = anchor_text.AnchorText(nlp, ['positive', 'negative'], use_unk_distribution=False)


# In[11]:


anchor_examples = [example for example in dev if len(example) < 100 and len(example)>15]


# In[21]:


pickle.dump( dev, open( "results/trigger_test.pickle", "wb" ))
pickle.dump( dev_labels, open( "results/trigger_test_labels.pickle", "wb" ))


# In[ ]:


my_utils = TextUtils(anchor_examples, dev, explainer, predict_sentences, "results/trigger_exps.pickle")
explanations = my_utils.compute_explanations(list(range(len(anchor_examples))))


# In[ ]:


pickle.dump( explanations, open( "results/trigger_exps_list.pickle", "wb" ))

