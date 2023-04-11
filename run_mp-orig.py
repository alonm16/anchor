#!/usr/bin/env python
# coding: utf-8

import warnings
import spacy
from old.modified_anchor import anchor_text, anchor_base
import pickle
import myUtils
from myUtils import *
from models.utils import *
from models.dataset_loader import *
import datetime
import time
import argparse
import os
import sys
sys.path.append('models')
# when apply torchscript to models sometimes
torch._C._jit_set_texpr_fuser_enabled(False)
from torch.multiprocessing import Pool, Process, set_start_method, SimpleQueue
from multiprocessing.managers import BaseManager
import copy

def process_compute(seed, anchor_examples, ignored, delta, dataset_name, model_type, i, indices, tokenizer, optimize, path, result_queue):
    
    torch._C._jit_set_texpr_fuser_enabled(False)
    anchor_text.AnchorText.set_optimize(optimize)
    
    nlp = spacy.load('en_core_web_sm')    
    device = torch.device(f'cuda:{i}')
    explainer = anchor_text.AnchorText(nlp, ['positive', 'negative'], use_unk_distribution=False)
    my_utils = TextUtils(anchor_examples, explainer, myUtils.old_predict_sentences, ignored, optimize = optimize, delta = delta)

    model = load_model(f'models/{model_type}/{dataset_name}/traced_{i}.pt').to(device)
    myUtils.model = model
    myUtils.tokenizer = tokenizer
    myUtils.device = device
    
    set_seed(seed)
    
    explanations = my_utils.compute_explanations(indices)
    result_queue.put(explanations)

def run():
    parser = argparse.ArgumentParser()
    torch.multiprocessing.freeze_support()
    set_start_method("spawn", force=True)
    
    num_processes = torch.cuda.device_count()
    warnings.simplefilter("ignore")
    sort_functions = {'polarity': sort_polarity, 'confidence': sort_confidence}
    
    parser.add_argument("--dataset_name", default='sentiment', choices = ['sentiment', 'corona', "dilemma", 'toy-spam', 'home-spam', 'sport-spam'])
    parser.add_argument("--model_type", default = 'tinybert', choices = ['tinybert', 'gru', 'svm', 'logistic'])
    parser.add_argument("--sorting", default='confidence', choices=['polarity', 'confidence'])
    parser.add_argument("--examples_max_length", default=200, type=int)
    parser.add_argument("--delta", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    examples_max_length = args.examples_max_length
    sort_function = sort_functions[args.sorting] 

    dataset_name = args.dataset_name
    sorting = args.sorting
    seed = args.seed
    model_type = args.model_type
    model_name = 'huawei-noah/TinyBERT_General_4L_312D'
    path = f'results/mp/{model_type}/{dataset_name}/{sorting}/{seed}/original'
    
    ds = get_ds(dataset_name)
        
    device = torch.device(f'cuda')
    model = load_model(f'models/{model_type}/{dataset_name}/traced.pt').to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
    myUtils.model = model
    myUtils.tokenizer = tokenizer
    myUtils.device = device
    
    anchor_examples, true_labels = preprocess_examples(ds, examples_max_length)
    anchor_examples, _ = sort_function(anchor_examples, true_labels)
    torch.cuda.empty_cache()

    if not os.path.exists(path):
        os.makedirs(path)
        
    ignored = []

    print(path)
    print(datetime.datetime.now())
    
    optimize = False
    
    processes = []
    indices = list(range(len(anchor_examples)))
    indices_list = np.array_split(indices, num_processes)
    
    result_queue = SimpleQueue()
    
    with CustomManager() as manager:

        for i in range(num_processes):
            p = torch.multiprocessing.Process(target=process_compute, args=([seed, anchor_examples, ignored, args.delta, dataset_name, model_type, i, indices_list[i], tokenizer, optimize, path, result_queue]))
            processes.append(p)

        st = time.time()

        for p in processes:
            p.start()

        explanations = []
        for _ in range(num_processes):
            explanations.extend(result_queue.get())

        for p in processes:
            p.join()
        
    pickle.dump(anchor_examples, open(f"{path}/anchor_examples.pickle", "wb"))
    pickle.dump(explanations, open(f"{path}/exps_list.pickle", "wb"))


    from csv import writer
    with open('times.csv', 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow([path[len('results/'):], (time.time()-st)/60, do_ignore, topk_optimize, desired_optimize ,examples_max_length])

            
class CustomManager(BaseManager):
    pass

if __name__ == '__main__':
    run()
