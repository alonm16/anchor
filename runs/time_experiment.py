#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import os.path
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble
import spacy
import sys
sys.path.append('..')
from sklearn.feature_extraction.text import CountVectorizer
from anchor import anchor_text
import time
from myUtils import *
import pickle
import matplotlib.pyplot as plt
import csv


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


vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
train_vectors = vectorizer.transform(train)
test_vectors = vectorizer.transform(test)
val_vectors = vectorizer.transform(val)


# In[6]:


c = sklearn.linear_model.LogisticRegression()
# c = sklearn.ensemble.RandomForestClassifier(n_estimators=500, n_jobs=10)
c.fit(train_vectors, train_labels)
preds = c.predict(val_vectors)
print('Val accuracy', sklearn.metrics.accuracy_score(val_labels, preds))
def predict_lr(texts):
    return c.predict(vectorizer.transform(texts))


# In[7]:


explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=False)


# In[8]:


train.sort(key=lambda example: len(example))


# In[14]:


with open("times.csv", "w") as time_file:
    writer = csv.writer(time_file)
    for i in range(0, len(train), 100):
        print(i)
        b = time.time()
        exp = explainer.explain_instance(train[i], predict_lr, threshold=0.95, verbose=False)
        writer.writerow(i, len(train[i]), time.time() - b)
        time_file.flush()


# In[58]:


with open('times.csv', newline='') as f:
    reader = csv.reader(f)
    rows = list(reader)
indices = [row[0] for row in rows if len(row)>0]
texts_length = [int(row[1]) for row in rows if len(row)>0]
predictions_time = [float(row[2]) for row in rows if len(row)>0]


# In[59]:


plt.scatter(texts_length, predictions_time, s = range(len(predictions_time)), alpha = 0.5)
plt.xlabel('texts_length')
plt.ylabel('predictions_time')
plt.title('text time anchor experiment')
plt.savefig("results/time.png")


# In[60]:


img = plt.imread("results/time.png")
plt.figure(figsize = (10,10))
plt.axis('off')
_ = plt.imshow(img)


# In[ ]:




