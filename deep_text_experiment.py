#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import os.path
import numpy as np
import sklearn
import spacy
import sys
from anchor import anchor_text
from myUtils import *
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from simpletransformers.classification import ClassificationModel


# In[2]:


# dataset from http://www.cs.cornell.edu/people/pabo/movie-review-data/
# Link: http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
def load_polarity(path='sentiment-sentences'):
    data = []
    labels = []
    f_names = ['rt-polarity.neg', 'rt-polarity.pos']
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), 'rb'):
            try:
                line.decode('utf8')
            except:
                continue
            data.append(line.strip())
            labels.append(l)
    return data, labels


# In[3]:


nlp = spacy.load('en_core_web_sm')


# In[4]:


data, labels = load_polarity()
train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)


# In[5]:


train = [example.decode('utf-8') for example in train]
val = [example.decode('utf-8') for example in val]
test = [example.decode('utf-8') for example in test]


# In[6]:


train_df = pd.DataFrame(list(zip(train,train_labels)))
train_df.columns = ["text", "labels"]

eval_df = pd.DataFrame(list(zip(val, val_labels)))
eval_df.columns = ["text", "labels"]

test_df = pd.DataFrame(list(zip(test,test_labels)))
test_df.columns = ["text", "labels"]


# In[7]:


def get_and_train():
    model = ClassificationModel('bert', 'bert-base-uncased', num_labels = 2)
    model.args.num_train_epochs = 2
    model.train_model(train_df)
    return model


# In[8]:


load_trained = True


# In[9]:


model = ClassificationModel('bert',  'outputs') if load_trained else get_and_train()


# In[10]:


def predict_lr(texts):
    return np.array(model.predict(texts)[0])


# In[11]:


explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=False)


# In[12]:


anchor_examples = [example for example in train if 20< len(example) < 70 and len(example)>20][:500]


# In[13]:


pickle.dump( test, open( "results/text_test_deep.pickle", "wb" ))
pickle.dump( test_labels, open( "results/text_test_labels_deep.pickle", "wb" ))


# In[ ]:


my_utils = TextUtils(anchor_examples, test, explainer, predict_lr, "results/text_exps_bert.pickle")
explanations = my_utils.compute_explanations(list(range(len(anchor_examples))))


# In[ ]:


pickle.dump( explanations, open( "results/text_exps_bert_list.pickle", "wb" ))

