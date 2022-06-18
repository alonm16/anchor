#!/usr/bin/env python
# coding: utf-8

# ## The modified algorithm returns the score for each word in the sentence instead of the best one, has option to ignore specific words like stop words of characters. in this notebook summing the scores bigger than the threshold (0.95) and dividing by the number of occurences. 

# In[3]:


# Setup
import matplotlib.pyplot as plt
import warnings
import spacy
from modified_anchor import anchor_text
import pickle
from myUtils import *
from transformer.utils import *
from dataset.dataset_loader import *
import datetime
import re

SEED = 84
torch.manual_seed(SEED)
warnings.simplefilter("ignore")


# In[4]:


plt.rcParams['font.size'] = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[5]:


# can be sentiment/spam/offensive
dataset_name = 'sentiment'
review_parser, label_parser, ds_train, ds_val, _ = create_sentiment_dataset()


# In[6]:


model = load_model('gru' , f'transformer/{dataset_name}/gru.pt', review_parser)
model = torch.jit.script(model)


# In[7]:

spacy_tokenizer = spacy.load("en_core_web_sm")

# 1 = pad 2=sos 3 = eos
def tokenize(text, max_len):
    sentence = spacy_tokenizer.tokenizer(text)
    input_tokens = [2] + [review_parser.vocab.stoi[word.text] for word in sentence] + [3] + [1]*(max_len-len(sentence))

    return input_tokens


# In[8]:

def predict_sentences(sentences):
    half_length = len(sentences)//2
    if(half_length>100):
        return np.concatenate([predict_sentences(sentences[:half_length]), predict_sentences(sentences[half_length:])])
    max_len = max([len(sentence) for sentence in sentences])
    sentences = torch.tensor([tokenize(sentence, max_len) for sentence in sentences], device=device)
    input_tokens = torch.transpose(sentences, 0, 1)
    output = model(input_tokens)

    return torch.argmax(output, dim=1).cpu().numpy()


# # Anchor Part

# In[9]:


nlp = spacy.load('en_core_web_sm')


# In[10]:


explainer = anchor_text.AnchorText(nlp, ['positive', 'negative'], use_unk_distribution=False)


# In[11]:

train, train_labels = [re.sub('\s+',' ',' '.join(example.text)) for example in ds_train], [example.label for example in ds_train]
test, test_labels = [re.sub('\s+',' ',' '.join(example.text)) for example in ds_train], [example.label for example in ds_train]

# In[12]:


anchor_examples = [example for example in train if len(example) < 90 and len(example)>20]


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


# ## notice!

# In[14]:


ignored = []


# In[ ]:
print(datetime.datetime.now())

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
my_utils = TextUtils(anchor_examples, test, explainer, predict_sentences, ignored,f"profile.pickle", optimize = True)
set_seed()
my_utils.compute_explanations(list(range(len(anchor_examples))))


# In[ ]:


print(datetime.datetime.now())

