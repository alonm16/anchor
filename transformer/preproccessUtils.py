from collections import Counter, defaultdict
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
from transformer.utils import *

def get_occurences(ds):
    c = Counter()
    ignore = list(".,- \'\"\s[]?():!;")
    ignore.extend(["--", "'s", 'sos', 'eos'])
    ignore.extend(stopwords.words('english'))
    for ds_example in ds:
        c.update(ds_example.text)
    for ignore_s in ignore:
        del c[ignore_s]
    values = np.array(list(c.values())).reshape(-1,1)
    scaler = MinMaxScaler().fit(values)
    for word in c.keys():
        c[word] = scaler.transform([[c[word]]])[0][0] + 1e-5
    return c

def get_label_distribution(ds):
    c_pos = Counter()
    c_neg = Counter()
    c = Counter()
    ignore = list(".,- \'\"\s[]?():!;")
    ignore.extend(["--", "'s", 'sos', 'eos'])
    ignore.extend(stopwords.words('english'))
    
    for ds_example in ds:
        if ds_example.label == 'positive':
            c_pos.update(set(ds_example.text))
        else:
            c_neg.update(set(ds_example.text))

    all_words = list(c_pos.keys())
    all_words.extend(c_neg.keys())
    all_words = set(all_words)
    for word in all_words:
        c[word] = (c_pos[word]-c_neg[word])/(c_pos[word]+c_neg[word])
    for ignore_s in ignore:
        c[ignore_s]=0
    
    return c

def get_prediction_distribution(ds, predictions):
    d = defaultdict(list)
    ignore = list(".,- \'\"\s[]?():!;")
    ignore.extend(["--", "'s", 'sos', 'eos'])
    ignore.extend(stopwords.words('english'))
    
    for ds_example, prediction in zip(ds, predictions):
        for word in ds_example.text:
            d[word].append(prediction)

    prediction_dict = defaultdict(lambda: 0.0)
    
    for word, values in d.items():
        prediction_dict[word] = sum(values)/len(values)
    for ignore_s in ignore:
        prediction_dict[ignore_s]=0.0
    
    return prediction_dict

import copy
def generate_example(ds, anchor_examples, explanations):
    
    filtered_examples = []
    for example in ds:
        for exp in explanations:     
            if ' '.join(example.text)==anchor_examples[exp.index]:
                c_example = copy.deepcopy(example)
                if exp.test_precision < 0.5:
                    c_example.label = 0
                else:
                    c_example.label = 2*(exp.test_precision - 0.5)
             
                filtered_examples.append(c_example)
                
                break
                
    return filtered_examples

import tqdm
import sys
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

def get_classes_for_csv():
    classes = None
    if type_dataset == 'trinary':
        classes = ['positive', 'negative', 'neutral']
    elif type_dataset == 'binary':
        classes = ['positive', 'negative']
    elif type_dataset == 'fine_grained':
        classes = ['positive', 'negative', 'neutral', 'very_positive', 'very_negative']
    index_csv = [f'pred_{curr_class}' for curr_class in classes]
    columns_csv = [f'real_{curr_class}' for curr_class in classes]
    return classes, index_csv, columns_csv

import pandas as pd
def compute_confusion_matrix(y_real, y_pred, model_name, type_dl):
    classes, index_csv, columns_csv = get_classes_for_csv()
    num_classes = len(classes)
    num_classes_in_y = len(torch.unique(y_real))
    assert num_classes == num_classes_in_y, 'Mismatch in number of classes'
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for class_index, class_name in enumerate(classes):
            all_pred_classes = y_pred[y_real == class_index]
            curr_col = torch.histc(all_pred_classes, bins=num_classes, min=0, max=num_classes - 1)
            confusion_matrix[:, class_index] = curr_col
    confusion_matrix = confusion_matrix.numpy()
    df = pd.DataFrame(confusion_matrix, index = index_csv, columns = columns_csv, dtype=int)
    #df.to_csv(output_directory / str('confusion_matrix_' + model_name + '_' + type_dl + '.csv'))
    return df
