#!/bin/sh

python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.1
python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.15 
python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.2
python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.35 
python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.5
python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.6 
python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.7
python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --optimization lossy
python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --optimization topk
python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --optimization desired

python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.1
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.15 
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.2
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.35 
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.5
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.6 
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.7
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --optimization lossy
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --optimization topk
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --optimization desired

python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.1
python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.15 
python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.2
python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.35 
python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.5
python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.6 
python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.7
python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --optimization lossy
python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --optimization topk
python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --optimization desired
