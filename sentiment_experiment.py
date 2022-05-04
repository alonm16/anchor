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


dataset_name = 'sentiment'
review_parser, label_parser, ds_train, ds_val, _ = create_sentiment_dataset()


# In[5]:


model = load_model('gru' , f'transformer/{dataset_name}/gru.pt', review_parser)


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


anchor_examples = [example for example in train if len(example) < 90 and len(example)>20]


# In[12]:


len(anchor_examples)


# In[13]:


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


# In[14]:


ignored = get_ignored(anchor_examples)


# In[15]:


##### notice!!!!!
ignored = []


# In[17]:


pickle.dump( test, open(f"{dataset_name}/test.pickle", "wb" ))
pickle.dump( test_labels, open( f"{dataset_name}/test_labels.pickle", "wb" ))
pickle.dump( anchor_examples, open( f"{dataset_name}/anchor_examples.pickle", "wb" ))


# In[18]:


print(datetime.datetime.now())


# In[ ]:


my_utils = TextUtils(anchor_examples, test, explainer, predict_sentences, ignored,f"{dataset_name}/exps.pickle")
explanations = my_utils.compute_explanations(list(range(len(anchor_examples))))


# In[ ]:


print(datetime.datetime.now())


# In[ ]:


pickle.dump( explanations, open( f"{dataset_name}/exps_list.pickle", "wb" ))


# # Loading Results

# In[10]:


test = np.array(pickle.load( open(  f"{dataset_name}/test.pickle", "rb" )))
test_labels = np.array(pickle.load( open(  f"{dataset_name}/test_labels.pickle", "rb" )))

explanations  = pickle.load(open(  f"{dataset_name}/exps_list.pickle", "rb" ))
anchor_examples = pickle.load( open(  f"{dataset_name}/anchor_examples.pickle", "rb" ))


# In[11]:


len(anchor_examples)


# In[12]:


len(explanations)


# In[13]:


test_predictions = np.array([predict_sentences([text])[0] for text in test])


# In[14]:


explanations = [ExtendedExplanation(exp, anchor_examples, test, test_labels, test_predictions ,predict_sentences, explainer) for exp in explanations if len(exp.fit_examples) > 0]
pickle.dump( explanations, open(  f"{dataset_name}/extended_exps.pickle", "wb" ))


# In[15]:


len(explanations)


# In[20]:


explanations = pickle.load(open(  f"{dataset_name}/extended_exps.pickle", "rb" ))






