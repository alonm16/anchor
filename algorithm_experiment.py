#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Setup

import matplotlib.pyplot as plt
import warnings
import spacy
from orig_anchor import anchor_text
import pickle
from myUtils import *
from transformer.utils import *
from dataset.dataset_loader import *
import datetime

SEED = 84
torch.manual_seed(SEED)
warnings.simplefilter("ignore")


# In[3]:


plt.rcParams['font.size'] = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[4]:


review_parser, label_parser, ds_train, ds_valid, ds_test = create_sentiment_dataset()
counter_test, counter_test_labels = counter_test()


# In[5]:


model = load_model('gru' , 'transformer/gru_sentiment.pt', review_parser)


# In[6]:


# 1 = pad 2=sos 3 = eos
def tokenize(text, max_len):
    sentence = review_parser.tokenize(str(text))
    input_tokens = [2] + [review_parser.vocab.stoi[word] for word in sentence] + [3] + [1]*(max_len-len(sentence))

    return input_tokens


# In[7]:


def predict_sentences(sentences):
    half_length = len(sentences)//2
    if(half_length>100):
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


# In[10]:


train, train_labels = [' '.join(example.text) for example in ds_train], [example.label for example in ds_train]
test, test_labels = [' '.join(example.text) for example in ds_train], [example.label for example in ds_train]


# In[11]:


anchor_examples = [example for example in train if len(example) < 70 and len(example)>20]


# In[73]:


from collections import Counter, defaultdict
from nltk.corpus import stopwords
def get_ignored(anchor_sentences):
    sentences = [[x.text for x in nlp(sentence)] for sentence in anchor_sentences]
    min_occurence = 1
    c = Counter()
    stop_words = list(".,- \'\"\s[]?():!;")
    stop_words.extend(["--", "'s", 'sos', 'eos'])
    stop_words.extend(stopwords.words('english'))
    """
    for sentence in sentences:
        c.update(sentence)
    sums = 0
    for ignore_s in stop_words:
        sums+=c[ignore_s]
        del c[ignore_s]
    print(sums)
    ignored_anchors = stop_words
    for key in c.keys():
        if c[key]<=min_occurence:
            ignored_anchors.append(key)
    print(len(c.keys()))
    return ignored_anchors
    """
    return stop_words


# In[75]:


ignored = get_ignored(anchor_examples)


# In[21]:


pickle.dump( test, open( "results/transformer_test.pickle", "wb" ))
pickle.dump( test_labels, open( "results/transformer_test_labels.pickle", "wb" ))
pickle.dump( anchor_examples, open( "results/transformer_anchor_examples.pickle", "wb" ))


# In[77]:


print(datetime.datetime.now())


# In[ ]:


my_utils = TextUtils(anchor_examples, counter_test, explainer, predict_sentences, ignored,"results/nostopwords_exps.pickle")
explanations = my_utils.compute_explanations(list(range(len(anchor_examples))))


# In[ ]:


print(datetime.datetime.now())


# In[ ]:


pickle.dump( explanations, open( "results/nostopwords_exps_list.pickle", "wb" ))


