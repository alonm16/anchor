#!/bin/sh 

for ds in "corona"; do
    for seed_val in 84 126 168 210; do
		python run.py --model_type tinybert --dataset_name $ds --sorting confidence --seed $seed_val --delta 0.7
		for delta_val in 0.5; do
            python run.py --model_type tinybert --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization masking_50
        done
    done
done

python run.py --model_type tinybert --dataset_name corona --sorting confidence --seed 42 --delta 0.7

for ds in "dilemma"; do
    for seed_val in 42 84 126 168 210; do
		python run.py --model_type tinybert --dataset_name $ds --sorting confidence --seed $seed_val --delta 0.7
		for delta_val in 0.1 0.15 0.35 0.5; do
            python run.py --model_type tinybert --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization masking_50
            python run.py --model_type tinybert --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization lossy topk
        done
    done
done

# for "lossy" "topk" "desired"; do
# 	for opt in delta_val in 0.1 0.15 0.2 0.35 0.5; do
# 		for seed_val in 42 84 126 168 210; do
# 			python run.py --model_type tinybert --dataset_name dilemma --sorting confidence --seed $seed_val --delta $delta_val --optimization $opt
# 		done
# 	done
# done

# for s in 10; do
# 	for d in results1/tinybert/corona/confidence/seed/$s/* ; do
# 		mkdir --parents results1/tinybert/corona/confidence/$(basename $d)/seed-$s;
# 		for f in $d/*; do
# 			mv $f results1/tinybert/corona/confidence/$(basename $d)/seed-$s/$(basename $f)
# 		done
# 	done
# done