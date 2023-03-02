#!/bin/sh
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.1 --examples_max_length 120
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.15 --examples_max_length 120
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.2 --examples_max_length 120
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.35 --examples_max_length 120
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.5 --examples_max_length 120

python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --examples_max_length 120 --optimization topk
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --examples_max_length 120 --optimization lossy
python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --examples_max_length 120 --optimization desired
