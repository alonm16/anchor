#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import os.path
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble
import spacy
import sys
from sklearn.feature_extraction.text import CountVectorizer
from anchor import anchor_text
from myUtils import *
import pickle
import matplotlib.pyplot as plt


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


# Note: you must have spacy installed. Run:
# 
#         pip install spacy && python -m spacy download en_core_web_sm
# 
# If you want to run BERT, you have to install transformers and torch or tf: 
# 
#         pip install torch transformers spacy && python -m spacy download en_core_web_sm
#         

# In[3]:


nlp = spacy.load('en_core_web_sm')


# In[5]:


data, labels = load_polarity()
train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)
counter_test, counter_test_labels = TextUtils.counter_test()


# In[6]:


vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
train_vectors = vectorizer.transform(train)
test_vectors = vectorizer.transform(test)
val_vectors = vectorizer.transform(val)
counter_test_vectors = vectorizer.transform(counter_test)


# In[10]:


c = sklearn.linear_model.LogisticRegression()
# c = sklearn.ensemble.RandomForestClassifier(n_estimators=500, n_jobs=10)
c.fit(train_vectors, train_labels)
preds = c.predict(val_vectors)
print('Val accuracy', sklearn.metrics.accuracy_score(val_labels, preds))
def predict_lr(texts):
    return c.predict(vectorizer.transform(texts))


# ## Using BERT

# In[11]:


explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=False)


# In[8]:


test = [example.decode('utf-8') for example in test]
anchor_examples = [example.decode('utf-8') for example in train]
anchor_examples = [example for example in anchor_examples if 20< len(example) < 70 and len(example)>20][:500]


# In[9]:


pickle.dump( test, open( "results/text_test.pickle", "wb" ))
pickle.dump( test_labels, open( "results/text_test_labels.pickle", "wb" ))


# In[ ]:


my_utils = TextUtils(anchor_examples, counter_test, explainer, predict_lr)
#explanations = my_utils.compute_explanations(np.random.choice(len(test), exps_num))
explanations = my_utils.compute_explanations(list(range(0, len(anchor_examples))))


# In[ ]:


pickle.dump( explanations, open( "results/text_exps.pickle", "wb" ))

