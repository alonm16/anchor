#!/bin/sh

python run.py --model_type tinybert --dataset_name sentiment --sorting confidence --delta 0.7
python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.7 --examples_max_length 120