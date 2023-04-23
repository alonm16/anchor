#!/usr/bin/env python
# coding: utf-8

import warnings
import spacy
from optimized_anchor import anchor_text, anchor_base
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

def process_compute(seed, anchor_examples, ignored, delta, dataset_name, model_type, i, indices, tokenizer, optimize, path, topk_optimize, desired_optimize, num_unmask, bg, result_queue):
    
    torch._C._jit_set_texpr_fuser_enabled(False)
    anchor_text.AnchorText.set_optimize(optimize)
    
    nlp = spacy.load('en_core_web_sm')    
    device = torch.device(f'cuda:{i}')
 
    explainer = anchor_text.AnchorText(nlp, ['positive', 'negative'], use_unk_distribution=False, device=device, num_unmask=num_unmask)
    my_utils = TextUtils(anchor_examples, explainer, myUtils.predict_sentences, ignored, optimize = optimize, delta = delta)
    
    anchor_base.AnchorBaseBeam.best_group = bg
    if model_type=='logistic':
        device = torch.device('cpu')
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
    parser.add_argument("--optimization", default='', choices = ['', 'topk', 'stop-words', 'desired', 'masking'], nargs = '+')
    parser.add_argument("--examples_max_length", default=200, type=int)
    parser.add_argument("--delta", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    examples_max_length = args.examples_max_length
    do_ignore = 'stop-words' in args.optimization
    min_value = 5
    topk_optimize = 'topk' in args.optimization
    desired_optimize = 'desired' in args.optimization
    num_unmask = 50 if 'masking' in args.optimization else 500
    sort_function = sort_functions[args.sorting] 

    dataset_name = args.dataset_name
    sorting = args.sorting
    seed = args.seed
    optimization = '-'.join(args.optimization)
    optimization = '-'.join([optimization, str(args.delta)]) if args.optimization!='' else args.delta
    model_type = args.model_type
    model_name = 'huawei-noah/TinyBERT_General_4L_312D'
    path = f'results2/mp/{model_type}/{dataset_name}/{sorting}/{seed}/{optimization}'
    
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

    if do_ignore:
        ignored = get_ignored(anchor_examples, min_value)
    else:
        ignored = []

    print(path)
    print(datetime.datetime.now())
    
    optimize = True
    
    processes = []
    indices = list(range(len(anchor_examples)))
    indices_list = np.array_split(indices, num_processes)
    
    result_queue = SimpleQueue()
    
    normal_occurences = get_occurences(anchor_examples)
    CustomManager.register('bg', BestGroup)
    
    with CustomManager() as manager:
        # create a shared bg instance
        bg = manager.bg(path, normal_occurences, filter_anchors = topk_optimize, desired_optimize = desired_optimize)

        for i in range(num_processes):
            p = torch.multiprocessing.Process(target=process_compute, args=([seed, anchor_examples, ignored, args.delta, dataset_name, model_type, i, indices_list[i], tokenizer, optimize, path, topk_optimize, desired_optimize, num_unmask, bg, result_queue]))
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
