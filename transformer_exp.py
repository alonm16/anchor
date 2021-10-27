#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Setup

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


# In[2]:


plt.rcParams['font.size'] = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[3]:


review_parser = None
label_parser = None
ds_train = None
ds_valid = None
ds_test = None

review_parser, label_parser, ds_train, ds_valid, ds_test = create_sentiment_dataset()


# ## Forward Function For Getting Accuracy

# In[4]:


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


# In[9]:


model = load_model('gru' , 'transformerUtils/gru_sentiment.pt', review_parser)


# In[10]:


# 1 = pad 2=sos 3 = eof 
def tokenize(text, max_len):
    sentence = review_parser.tokenize(str(text))
    input_tokens = [2] + [review_parser.vocab.stoi[word] for word in sentence] + [3] + [1]*(max_len-len(sentence))

    return input_tokens


# In[11]:


def predict_sentences(sentences):
    half_length = len(sentences)//2
    if(half_length>800):
        return np.concatenate([predict_sentences(sentences[:half_length]), predict_sentences(sentences[half_length:])])
    max_len = max([len(sentence) for sentence in sentences])
    sentences = torch.tensor([tokenize(sentence, max_len) for sentence in sentences]).to(device)
    input_tokens = torch.transpose(sentences, 0, 1)
    output = model(input_tokens)
    return torch.argmax(output, dim=1).cpu().numpy()


# # Anchor Part

# In[12]:


nlp = spacy.load('en_core_web_sm')


# In[14]:


explainer = anchor_text.AnchorText(nlp, ['positive', 'negative'], use_unk_distribution=False)


# In[13]:


train, train_labels = [' '.join(example.text) for example in ds_train], [example.label for example in ds_train]
test, test_labels = [' '.join(example.text) for example in ds_train], [example.label for example in ds_train]


# In[18]:


anchor_examples = [example for example in train if len(example) < 70 and len(example)>20]


# In[21]:


pickle.dump( test, open( "results/transformer_test.pickle", "wb" ))
pickle.dump( test_labels, open( "results/transformer_test_labels.pickle", "wb" ))


# In[19]:


my_utils = TextUtils(anchor_examples, test, explainer, predict_sentences, "results/transformer_exps.pickle")
explanations = my_utils.compute_explanations(list(range(len(anchor_examples))))


# In[ ]:


pickle.dump( explanations, open( "results/transformer_exps_list.pickle", "wb" ))


# # Training Function
# 
# ### Saves all the the output in the output directory

# In[5]:


import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
def train_model(model_name, device, output_directory = 'results'):
    NUM_EPOCHS = 100
    if model_name == 'gru':
        hp = load_hyperparams(model_name, type_dataset)
        model = models.VanillaGRU(review_parser.vocab, hp['embedding_dim'], hp['hidden_dim'], hp['num_layers'], hp['output_classes'], hp['dropout']).to(device)
    elif model_name == 'attention':
        hp = load_hyperparams(model_name, type_dataset)
        model = models.MultiHeadAttentionNet(input_vocabulary=review_parser.vocab, embed_dim=hp['embedding_dim'], num_heads=hp['num_heads'], 
                                           dropout=hp['dropout'], two_attention_layers=hp['two_atten_layers'], output_classes=hp['output_classes']).to(device)
    print(model)
    dl_train, dl_valid, dl_test = torchtext.legacy.data.BucketIterator.splits((ds_train, ds_valid, ds_test), batch_size=hp['batch_size'], sort = False, shuffle=True, device=device)

    optimizer = optim.Adam(model.parameters(), lr=hp['lr'])
    loss_fn = nn.NLLLoss()
    
    trainer = training.SentimentTrainer(model, loss_fn, optimizer, device)
    checkpoint_filename = str(output_directory) +  '/' + model_name
    print(f'Saving checkpoint with prefix: {checkpoint_filename}')
    fit_res = trainer.fit(dl_train, dl_valid, NUM_EPOCHS, early_stopping = hp['early_stopping'], checkpoints = checkpoint_filename, params = hp)

    fig, axes = plot.plot_fit(fit_res)
    fig.savefig(output_directory + '/' + str(model_name + '.png'))
    saved_state = torch.load(checkpoint_filename + '.pt', map_location=device)
    model.load_state_dict(saved_state["model_state"])
    loaded_hp = saved_state["parameters"]
    print('----- Loaded params ------')
    print(loaded_hp)
    all_dataloaders = [dl_train, dl_valid, dl_test]
    type_dls = ['train', 'valid', 'test']
    accuracies = []
    for dl, type_dl in zip(all_dataloaders, type_dls):
        y_real, y_pred, accuracy = forward_dl(model, dl, device, type_dl)
        df = compute_confusion_matrix(y_real, y_pred, model_name, type_dl)
        accuracies.append(accuracy)
        display(df)
    numpy_accuracy = np.array(accuracies)
    df = pd.DataFrame(numpy_accuracy, index = type_dls, dtype=float)
    df.to_csv(output_directory + '/' + str('accuracies_' + model_name + '.csv')) 


# # Training GRU

# In[7]:


SEED = 84
torch.manual_seed(SEED)
#train_model('gru', device)


# # Training Attention

# In[ ]:


SEED = 84
torch.manual_seed(SEED)
#train_model('attention', device)


# # Loading Results

# In[10]:


test = pickle.load( open( "results/transformer_test.pickle", "rb" ))
test_labels = pickle.load( open( "results/transformer_test_labels.pickle", "rb" ))
test = np.array(test)
test_labels = np.array(test_labels)
exps_file  = open( "results/transformer_exps.pickle", "rb" )
num_exps = 500
explanations= []


# In[11]:


for _ in range(num_exps):
    explanations.append(pickle.load(exps_file))
exps_file.close()


# In[12]:


myUtils =  TextUtils(test, test, None, None, "results/transformer_exps.pickle")
explanations = myUtils.remove_duplicates(explanations)
explanations = [exp for exp in explanations if len(exp.fit_examples) > 10]
explanations.sort(key=lambda exp: exp.precision)


# In[13]:


best = explanations[-10:]
best.reverse()
for exp in best:
    print("------------------------")
    exp_label =  predict_sentences([str(test[exp.index])])[0]
    print('Prediction:', explainer.class_names[exp_label])
    print('Anchor: %s' % (' AND '.join(exp.names)))
    print('Precision: %.2f' % exp.precision)
    print('Coverage: %.2f' % exp.coverage)
    print('Anchor test precision: %.2f' % (np.mean(predict_sentences(str(test[exp.fit_examples])) == exp_label)))
    print('Anchor test coverage: %.2f' % (exp.test_cov))
    covered_labels = test_labels[exp.fit_examples]
    real_percentage = np.mean(covered_labels == explainer.class_names[exp_label])
    print('Anchor test REAL precision: %.2f' % real_percentage)


# In[14]:


exps = explanations[-len(explanations)//2:]
exps = [exp for exp in exps if len(exp.fit_examples)>0]
real_precisions = [np.mean(test_labels[exp.fit_examples] == explainer.class_names[predict_sentences([str(test[exp.index])])[0]])
                   for exp in exps]
test_precisions = [np.mean(predict_sentences(test[exp.fit_examples]) == predict_sentences([str(test[exp.index])])[0]) 
                   for exp in exps]


# In[15]:


plt.scatter(test_precisions, real_precisions, s = range(len(exps)), alpha = 0.5)
plt.xlabel('test precision')
plt.ylabel('real precision')
plt.title('text anchor experiment')
plt.savefig("results/transformer.png")


# In[16]:


img = plt.imread("results/transformer.png")
plt.figure(figsize = (10,10))
plt.axis('off')
_ = plt.imshow(img)


# In[ ]:




