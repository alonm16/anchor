#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Setup
import os
import sys
import time
import torch
import matplotlib.pyplot as plt
import warnings
import torch.nn as nn
import spacy
from anchor import anchor_text
import pickle
from myUtils import *
import transformerUtils.models as models

SEED = 84
torch.manual_seed(SEED)
warnings.simplefilter("ignore")


# In[2]:


plt.rcParams['font.size'] = 20
data_dir = os.path.expanduser('~/.pytorch-datasets')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[3]:


type_dataset = 'binary' # What kind of dataset to train on trinary, fine_grained or binary


# # Preparing the Datasets

# In[4]:


import torchtext.data
import torchtext.datasets


# In[5]:


def load_dataset(fine_grained = False):
    # torchtext Field objects parse text (e.g. a review) and create a tensor representation

    # This Field object will be used for tokenizing the movie reviews text
    # For this application, tokens ~= words

    review_parser = torchtext.data.Field(
        sequential=True, use_vocab=True, lower=True, dtype=torch.long,
        tokenize='spacy', tokenizer_language='en_core_web_sm', init_token='sos', eos_token='eos'
    )

    # This Field object converts the text labels into numeric values (0,1,2)
    label_parser = torchtext.data.Field(
        is_target=True, sequential=False, unk_token=None, use_vocab=True
    )
    
    # Load SST, tokenize the samples and labels
    # ds_X are Dataset objects which will use the parsers to return tensors
    ds_train, ds_valid, ds_test = torchtext.datasets.SST.splits(
        review_parser, label_parser, root=data_dir, fine_grained=fine_grained
    )


    return review_parser, label_parser, ds_train, ds_valid, ds_test


# In[6]:


def build_vocabulary(review_parser, label_parser, ds_train, min_freq):
    review_parser.build_vocab(ds_train, min_freq = min_freq)
    label_parser.build_vocab(ds_train, min_freq= min_freq)

    print(f"Number of tokens in training samples: {len(review_parser.vocab)}")
    print(f"Number of tokens in training labels: {len(label_parser.vocab)}")
    return review_parser, label_parser


# In[7]:


def print_dataset(ds, classes, type_ds):
    print(f'Number of {type_ds} samples: {len(ds)}') 
    for class_type in classes:
        number_of_class = len([example for example in ds if example.label == class_type])
        print(f'Number of {class_type} examples {number_of_class}')
    print('------------------------------------------------------')


# # Creating Binary Dataset

# In[8]:


def filter_neutral(ds):
    ds.examples = [example for example in ds.examples if example.label != 'neutral']


# In[9]:


def create_binary_dataset():
    review_parser, label_parser, ds_train, ds_valid, ds_test = load_dataset(fine_grained = False)
    review_parser, label_parser = build_vocabulary(review_parser, label_parser, ds_train, min_freq=5)
    filter_neutral(ds_train)
    filter_neutral(ds_valid)
    filter_neutral(ds_test)
    print_dataset(ds_train, ['positive', 'negative'], 'training')
    print_dataset(ds_valid, ['positive', 'negative'], 'validation')
    print_dataset(ds_test, ['positive', 'negative'], 'test')
    return review_parser, label_parser, ds_train, ds_valid, ds_test


# In[10]:


review_parser = None
label_parser = None
ds_train = None
ds_valid = None
ds_test = None

review_parser, label_parser, ds_train, ds_valid, ds_test = create_binary_dataset()


# ## Forward Function For Getting Accuracy

# In[11]:


import tqdm
def forward_dl(model, dl, device, type_dl):
    model.train(False)
    num_samples = len(dl) * dl.batch_size
    num_batches = len(dl)  
    pbar_name = type(model).__name__
    list_y_real = []
    list_y_pred = []
    pbar_file = sys.stdout
    num_correct = 0
    dl_iter = iter(dl)
    for batch_idx in range(num_batches):
        data = next(dl_iter)
        x, y = data.text, data.label
        list_y_real.append(y)
        x = x.to(device)  # (S, B, E)
        y = y.to(device)  # (B,)
        with torch.no_grad():
            if isinstance(model, models.VanillaGRU):
                y_pred_log_proba = model(x)
            elif isinstance(model, models.MultiHeadAttentionNet):
                y_pred_log_proba, _ = model(x)
            y_pred = torch.argmax(y_pred_log_proba, dim=1)
            num_correct += torch.sum(y_pred == y).float().item()
            list_y_pred.append(y_pred)
    accuracy = 100.0 * num_correct / num_samples
    print(f'Accuracy for {type_dl} is {accuracy}')
    
    all_y_real = torch.cat(list_y_real)
    all_y_pred = torch.cat(list_y_pred)
    return all_y_real, all_y_pred, accuracy


# ## Loading Hyper Parameters

# In[12]:


import transformerUtils.hyperparams as hyperparams
def load_hyperparams(model_type, type_dataset):
    hp = None
    if model_type == 'gru':
        hp = hyperparams.hyperparams_for_gru_binary()
    elif model_type == 'attention':
        hp = hyperparams.hyperparams_for_attention_binary()
    print(hp)
    return hp


# # Load Model

# In[13]:


def load_model(model_name, path):
    if model_name == 'gru':
        hp = load_hyperparams(model_name, type_dataset)
        model = models.VanillaGRU(review_parser.vocab, hp['embedding_dim'], hp['hidden_dim'], hp['num_layers'], hp['output_classes'], hp['dropout']).to(device)
    elif model_name == 'attention':
        hp = load_hyperparams(model_name, type_dataset)
        model = models.MultiHeadAttentionNet(input_vocabulary=review_parser.vocab, embed_dim=hp['embedding_dim'], num_heads=hp['num_heads'], 
                                           dropout=hp['dropout'], two_attention_layers=hp['two_atten_layers'], output_classes=hp['output_classes']).to(device)
    saved_state = torch.load(path, map_location=device)
    model.load_state_dict(saved_state["model_state"])
    print(model)
    
    return model


# In[14]:


model = load_model('gru' , 'transformerUtils/gru.pt')


# In[15]:


# 1 = pad 2=sos 3 = eof 
def tokenize(text, max_len):
    sentence = review_parser.tokenize(text)
    input_tokens = [2] + [review_parser.vocab.stoi[word] for word in sentence] + [3] + [1]*(max_len-len(sentence))

    return input_tokens


# In[23]:


def predict_sentences(sentences):
    max_len = max([len(sentence) for sentence in sentences])
    sentences = torch.tensor([tokenize(sentence, max_len) for sentence in sentences]).to(device)
    input_tokens = torch.transpose(sentences, 0, 1)
    output = model(input_tokens)
    return torch.argmax(output, dim=1).cpu().numpy()


# # Anchor Part

# In[17]:


nlp = spacy.load('en_core_web_sm')


# In[18]:


explainer = anchor_text.AnchorText(nlp, ['positive', 'negative'], use_unk_distribution=False)


# In[25]:


train, train_labels = [' '.join(example.text) for example in ds_train], [example.label for example in ds_train]
test, test_labels = [' '.join(example.text) for example in ds_train], [example.label for example in ds_train]


# In[20]:


anchor_examples = [example for example in train if len(example) < 70 and len(example)>20][:500]


# In[21]:


pickle.dump( test, open( "results/transformer_test.pickle", "wb" ))
pickle.dump( test_labels, open( "results/transformer_test_labels.pickle", "wb" ))


# In[26]:


my_utils = TextUtils(anchor_examples, test, explainer, predict_sentences, "results/transformer_exps.pickle")
explanations = my_utils.compute_explanations(list(range(len(anchor_examples))))


# In[ ]:


pickle.dump( explanations, open( "results/transformer_exps_list.pickle", "wb" ))


# ## Training Function
# 
# ### Saves all the the output in the output directory

# In[ ]:


