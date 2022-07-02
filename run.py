#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Setup
import warnings
import spacy
from modified_anchor import anchor_text
import pickle
import myUtils
from myUtils import *
from transformer.utils import *
from dataset.dataset_loader import *
import datetime
get_ipython().run_line_magic('load_ext', 'line_profiler')

SEED = 84
torch.manual_seed(SEED)
warnings.simplefilter("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[2]:


# can be sentiment/spam/offensive
dataset_name = 'sentiment'
text_parser, label_parser, ds_train, ds_val = get_dataset(dataset_name)


# In[3]:


model = load_model('gru' , f'transformer/{dataset_name}/gru.pt', text_parser)
myUtils.model = torch.jit.script(model)
myUtils.text_parser = text_parser


# In[4]:


nlp = spacy.load('en_core_web_sm')


# In[5]:


train, train_labels, test, test_labels, anchor_examples = preprocess_examples(ds_train)


# In[6]:


ignored = get_ignored(anchor_examples)
normal_occurences = get_occurences(anchor_examples)
from modified_anchor import anchor_base
anchor_base.AnchorBaseBeam.best_group = BestGroup(normal_occurences)


# ## notice!

# In[7]:


ignored = []

# In[10]:


optimize = True
anchor_text.AnchorText.set_optimize(optimize)
explainer = anchor_text.AnchorText(nlp, ['positive', 'negative'], use_unk_distribution=False)


# In[16]:


# pickle.dump( test, open(f"{dataset_name}/test.pickle", "wb" ))
# pickle.dump( test_labels, open( f"{dataset_name}/test_labels.pickle", "wb" ))
# pickle.dump( anchor_examples, open( f"{dataset_name}/anchor_examples.pickle", "wb" ))
    
my_utils = TextUtils(anchor_examples, test, explainer, predict_sentences, ignored,f"profile.pickle", optimize = True)
set_seed()
explanations = my_utils.compute_explanations(list(range(len(anchor_examples))))

pickle.dump( explanations, open( f"{dataset_name}/profile_list.pickle", "wb"))


# In[ ]:


print(datetime.datetime.now())

