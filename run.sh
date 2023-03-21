#!/bin/sh 

for delta_val in 0.1 0.15 0.2 0.35 0.5; do
	for seed_val in 42 84 126 168 210; do
		python run_seed.py --model_type tinybert --dataset_name dilemma --sorting confidence --seed $seed_val --delta $delta_val
	done
done

# for s in 10; do
# 	for d in results1/tinybert/corona/confidence/seed/$s/* ; do
# 		mkdir --parents results1/tinybert/corona/confidence/$(basename $d)/seed-$s;
# 		for f in $d/*; do
# 			mv $f results1/tinybert/corona/confidence/$(basename $d)/seed-$s/$(basename $f)
# 		done
# 	done
# done