#!/bin/sh
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 42 --delta 0.1 --optimization lossy
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 84 --delta 0.15 --optimization lossy
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 126 --delta 0.2 --optimization lossy
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 168 --delta 0.35 --optimization lossy
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 210 --delta 0.5 --optimization lossy

python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 42 --delta 0.1 --optimization topk
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 84 --delta 0.15 --optimization topk
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 126 --delta 0.2 --optimization topk
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 168 --delta 0.35 --optimization topk
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 210 --delta 0.5 --optimization topk

python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 42 --delta 0.1 --optimization desired
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 84 --delta 0.15 --optimization desired
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 126 --delta 0.2 --optimization desired
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 168 --delta 0.35 --optimization desired
python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 210 --delta 0.5 --optimization desired
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.1
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.15 
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.2
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.35 
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.5
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.6 
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.7
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --optimization lossy
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --optimization topk
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --optimization desired

# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.1
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.15 
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.2
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.35 
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.5
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.6 
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.7
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --optimization lossy
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --optimization topk
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --optimization desired

# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.1
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.15 
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.2
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.35 
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.5
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.6 
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.7
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --optimization lossy
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --optimization topk
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --optimization desired
