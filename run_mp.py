#!/usr/bin/env python
# coding: utf-8

import warnings
import spacy
from optimized_anchor import anchor_text, anchor_base
import pickle
import myUtils
from myUtils import *
from models.utils import *
from dataset.dataset_loader import *
import datetime
import time
import argparse
import os
from models.utils import *
from torch.multiprocessing import Pool, Process, set_start_method, SimpleQueue
from transformers import AutoModelForSequenceClassification
import copy

def process_compute(anchor_examples, ignored, delta, dataset_name, model_type, test, i, indices, tokenizer, optimize, folder_name, normal_occurences, topk_optimize, desired_optimize, result_queue):
    anchor_text.AnchorText.set_optimize(optimize)
    
    nlp = spacy.load('en_core_web_sm')
    explainer = anchor_text.AnchorText(nlp, ['positive', 'negative'], use_unk_distribution=False)
    my_utils = TextUtils(anchor_examples, test, explainer, predict_sentences, ignored, f"profile-{dataset_name}-{i}.pickle", optimize = True, delta=delta)
    
    anchor_base.AnchorBaseBeam.best_group = BestGroup(folder_name, normal_occurences, filter_anchors = topk_optimize, desired_optimize = desired_optimize)
    
    model = torch.jit.load(f'models/{model_type}/{dataset_name}/traced.pt').to(device)
    model = model.eval()
    myUtils.model = model
    myUtils.tokenizer = tokenizer
    explanations = my_utils.compute_explanations(indices)
    result_queue.put(explanations)

def run():
    torch.multiprocessing.freeze_support()
    set_start_method("spawn", force=True)
    
    SEED = 84
    num_processes = 5
    torch.manual_seed(SEED)
    warnings.simplefilter("ignore")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    sort_functions = {'polarity': sort_polarity, 'confidence': sort_confidence}

    parser.add_argument("--dataset_name", default='sentiment', choices = ['sentiment', 'offensive', 'corona', 'sentiment_twitter'])
    parser.add_argument("--sorting", default='polarity', choices=['polarity', 'confidence'])
    parser.add_argument("--optimization", default='', choices = ['', 'topk', 'lossy', 'desired'])
    parser.add_argument("--examples_max_length", default=90, type=int)
    parser.add_argument("--delta", default=0.1, type=float)

    args = parser.parse_args()

    examples_max_length = args.examples_max_length
    do_ignore = args.optimization=='lossy'
    topk_optimize = args.optimization=='topk'
    desired_optimize = args.optimization=='desired'
    sort_function = sort_functions[args.sorting]
    delta = args.delta

    # can be sentiment/offensive/corona
    dataset_name = args.dataset_name
    sorting = args.sorting
    optimization = args.optimization
    model_type = 'tinybert'
    model_name = 'huawei-noah/TinyBERT_General_4L_312D'
    folder_name = f'results/mp/{dataset_name}/{sorting}/{optimization}' if optimization!='' else f'results/mp/{dataset_name}/{sorting}/{delta}'

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    _, label_parser, ds_train, ds_val = get_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
    model = torch.jit.load(f'models/{model_type}/{dataset_name}/traced.pt').to(device)
    model = model.eval()
    myUtils.model = model
    myUtils.tokenizer = tokenizer

    train, train_labels, test, test_labels, anchor_examples = preprocess_examples(ds_train, examples_max_length)

    anchor_examples = sort_function(anchor_examples)


    # In[6]:

    normal_occurences = get_occurences(anchor_examples)

    # In[7]:

    if do_ignore:
        ignored = get_ignored(anchor_examples)
    else:
        ignored = []

    print(folder_name)
    print(datetime.datetime.now())
    
    optimize = True
    anchor_text.AnchorText.set_optimize(optimize)
    
    processes = []
    indices = list(range(len(anchor_examples)))
    indices_list = np.array_split(indices, num_processes)
    
    result_queue = SimpleQueue()

    for i in range(num_processes):
        p = torch.multiprocessing.Process(target=process_compute, args=([anchor_examples, ignored, delta, dataset_name, model_type, test, i, indices_list[i], tokenizer, optimize, folder_name, normal_occurences, topk_optimize, desired_optimize, result_queue]))
        processes.append(p)
		
    st = time.time()

    for p in processes:
        set_seed()
        p.start()
        
    explanations = []
    for _ in range(num_processes):
        explanations.extend(result_queue.get())
        
    for p in processes:
        p.join()
        
    pickle.dump(anchor_examples, open(f"{folder_name}/anchor_examples.pickle", "wb"))
    pickle.dump(explanations, open(f"{folder_name}/exps_list.pickle", "wb"))

    print(datetime.datetime.now())

    from csv import writer
    with open('times.csv', 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow([folder_name, (time.time()-st)/60, do_ignore, topk_optimize, desired_optimize ,examples_max_length])
			

if __name__ == '__main__':
    run()
