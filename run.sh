#!/bin/bash
deltas=(0.1 0.15 0.2 0.35 0.5)
seeds=(42 84 126 168 210)
ds="toy-spam"

for delta_val in ${deltas[@]}; do
	for seed_val in ${seeds[@]}; do
		python run_mp.py --model_type tinybert --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val
	done
done

for opt in "stop-words" "topk" "desired" "masking"; do
	for delta_val in ${deltas[@]}; do
		for seed_val in ${seeds[@]}; do
			python run_mp.py --model_type tinybert --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization $opt
		done
	done
done

for delta_val in ${deltas[@]}; do
	for seed_val in ${seeds[@]}; do
		python run_mp.py --model_type tinybert --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization stop-words topk
	done
done

for delta_val in ${deltas[@]}; do
    for seed_val in ${seeds[@]}; do
        python run_mp.py --model_type tinybert --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization stop-words topk masking
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
