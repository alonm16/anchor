#!/bin/sh

python run.py --model_type tinybert --dataset_name spam --sorting confidence --delta 0.1 
python run.py --model_type tinybert --dataset_name spam --sorting confidence --delta 0.15 
python run.py --model_type tinybert --dataset_name spam --sorting confidence --delta 0.2 
python run.py --model_type tinybert --dataset_name spam --sorting confidence --delta 0.35 
python run.py --model_type tinybert --dataset_name spam --sorting confidence --delta 0.5 

python run.py --model_type tinybert --dataset_name spam --sorting confidence --optimization lossy 
python run.py --model_type tinybert --dataset_name spam --sorting confidence --optimization topk
python run.py --model_type tinybert --dataset_name spam --sorting confidence --optimization desired




