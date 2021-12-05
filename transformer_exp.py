#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import time
import matplotlib.pyplot as plt
import warnings
import torch.nn as nn
import spacy
from anchor import anchor_text
import pickle
from myUtils import *
import transformerUtils.models as models
import transformerUtils.training as training
import transformerUtils.plot as plot
from transformerUtils.utils import *

SEED = 84
torch.manual_seed(SEED)
warnings.simplefilter("ignore")


# In[3]:


plt.rcParams['font.size'] = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[4]:


review_parser = None
label_parser = None
ds_train = None
ds_valid = None
ds_test = None

review_parser, label_parser, ds_train, ds_valid, ds_test =  create_sentiment_dataset()
counter_test, counter_test_labels = counter_test()


# In[5]:


model = load_model('gru' , 'transformerUtils/gru_sentiment.pt', review_parser)


# In[6]:


# 1 = pad 2=sos 3 = eof 
def tokenize(text, max_len):
    sentence = review_parser.tokenize(str(text))
    input_tokens = [2] + [review_parser.vocab.stoi[word] for word in sentence] + [3] + [1]*(max_len-len(sentence))

    return input_tokens


# In[7]:


def predict_sentences(sentences):
    half_length = len(sentences)//2
    if(half_length>200):
        return np.concatenate([predict_sentences(sentences[:half_length]), predict_sentences(sentences[half_length:])])
    max_len = max([len(sentence) for sentence in sentences])
    sentences = torch.tensor([tokenize(sentence, max_len) for sentence in sentences]).to(device)
    input_tokens = torch.transpose(sentences, 0, 1)
    output = model(input_tokens)
    return torch.argmax(output, dim=1).cpu().numpy()


# # Anchor Part

# In[8]:


nlp = spacy.load('en_core_web_sm')


# In[9]:


explainer = anchor_text.AnchorText(nlp, ['positive', 'negative'], use_unk_distribution=False)


# In[13]:


train, train_labels = [' '.join(example.text) for example in ds_train], [example.label for example in ds_train]
test, test_labels = [' '.join(example.text) for example in ds_train], [example.label for example in ds_train]


# In[18]:


anchor_examples = pickle.load( open( "results/transformer_anchor_examples.pickle", "rb" ))


# In[21]:


#pickle.dump( test, open( "results/transformer_test.pickle", "wb" ))
#pickle.dump( test_labels, open( "results/transformer_test_labels.pickle", "wb" ))


# In[ ]:


my_utils = TextUtils(anchor_examples, counter_test, explainer, predict_sentences, "results/gru_counter_exps.pickle")
explanations = my_utils.compute_explanations(list(range(len(anchor_examples))))


# In[ ]:


pickle.dump( explanations, open( "results/gru_counter_exps_list.pickle", "wb" ))


