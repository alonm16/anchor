#!/bin/sh
python run.py --model_type tinybert --dataset_name corona --sorting confidence --delta 0.6
python run.py --model_type tinybert --dataset_name corona --sorting confidence --delta 0.7
python run.py --model_type tinybert --dataset_name corona --sorting confidence --delta 0.8

python run.py --model_type tinybert --dataset_name dilemma --sorting confidence --delta 0.6
python run.py --model_type tinybert --dataset_name dilemma --sorting confidence --delta 0.7
python run.py --model_type tinybert --dataset_name dilemma --sorting confidence --delta 0.8

python run.py --model_type tinybert --dataset_name sentiment --sorting confidence --delta 0.6
python run.py --model_type tinybert --dataset_name sentimnet --sorting confidence --delta 0.7
python run.py --model_type tinybert --dataset_name sentiment --sorting confidence --delta 0.8

python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.6 --examples_max_length 120
python run.py --model_type tinybert --dataset_name sporm-spam --sorting confidence --delta 0.7 --examples_max_length 120
python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.8 --examples_max_length 120