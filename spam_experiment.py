#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


plt.rcParams['font.size'] = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[3]:


review_parser, label_parser, ds_train, ds_val = spam_dataset()


# In[6]:


model = load_model('gru' , 'transformer/spam/gru.pt', review_parser)


# In[7]:


# 1 = pad 2=sos 3 = eos
def tokenize(text, max_len):
    sentence = review_parser.tokenize(str(text))
    input_tokens = [2] + [review_parser.vocab.stoi[word] for word in sentence] + [3] + [1]*(max_len-len(sentence))

    return input_tokens


# In[8]:


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

# In[9]:


nlp = spacy.load('en_core_web_sm')


# In[10]:


explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=False)


# In[11]:


train, train_labels = [' '.join(example.text) for example in ds_train], [example.label for example in ds_train]
test, test_labels = [' '.join(example.text) for example in ds_train], [example.label for example in ds_train]


# In[26]:


anchor_examples = [example for example in train if len(example) < 100 and len(example)>20]


# In[27]:


len(anchor_examples)





##### notice!!!!!
ignored = []


# In[23]:


pickle.dump( test, open( "spam/transformer_test.pickle", "wb" ))
pickle.dump( test_labels, open( "spam/transformer_test_labels.pickle", "wb" ))
pickle.dump( anchor_examples, open( "spam/transformer_anchor_examples.pickle", "wb" ))


# In[24]:


print(datetime.datetime.now())


# In[ ]:


my_utils = TextUtils(anchor_examples, test, explainer, predict_sentences, ignored,"spam/transformer_exps.pickle")
explanations = my_utils.compute_explanations(list(range(len(anchor_examples))))


# In[ ]:


print(datetime.datetime.now())


# In[ ]:


pickle.dump( explanations, open( "spam/transformer_exps_list.pickle", "wb" ))


