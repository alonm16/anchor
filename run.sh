#!/bin/sh
for s in 10; do
	for d in results1/tinybert/corona/confidence/seed/$s/* ; do
		mkdir --parents results1/tinybert/corona/confidence/$(basename $d)/seed-$s;
		for f in $d/*; do
			mv $f results1/tinybert/corona/confidence/$(basename $d)/seed-$s/$(basename $f)
		done
	done
done
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 500 --delta 0.5 --optimization topk

# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 126 --delta 0.1 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 168 --delta 0.1 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 210 --delta 0.1 --optimization topk

# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 42 --delta 0.15 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 84 --delta 0.15 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 126 --delta 0.15 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 168 --delta 0.15 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 210 --delta 0.15 --optimization topk

# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 42 --delta 0.2 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 84 --delta 0.2 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 126 --delta 0.2 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 168 --delta 0.2 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 210 --delta 0.2 --optimization topk

# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 42 --delta 0.35 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 84 --delta 0.35 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 126 --delta 0.35 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 168 --delta 0.35 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 210 --delta 0.35 --optimization topk

# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 42 --delta 0.5 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 84 --delta 0.5 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 126 --delta 0.5 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 168 --delta 0.5 --optimization topk
# python run_seed.py --model_type tinybert --dataset_name corona --sorting confidence --seed 210 --delta 0.5 --optimization topk

# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.15
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.155 
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.2
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.35 
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.5
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.6 
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --delta 0.7
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --optimization lossy
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --optimization topk
# python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --optimization desired

# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.15
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.155 
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.2
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.35 
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.5
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.6 
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --delta 0.7
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --optimization lossy
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --optimization topk
# python run.py --model_type tinybert --dataset_name home-spam --sorting confidence --optimization desired

# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.15
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.155 
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.2
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.35 
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.5
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.6 
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --delta 0.7
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --optimization lossy
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --optimization topk
# python run.py --model_type tinybert --dataset_name sport-spam --sorting confidence --optimization desired
