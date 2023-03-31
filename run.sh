#!/bin/sh 

deltas = (0.1 0.15 0.2 0.35 0.5)
seeds = (42 84 126 168 210)

for ds in "corona" "dilemma"; do
	for opt in "masking_50"; do
		for delta_val in ${deltas[@]}; do
			for seed_val in ${seeds[@]}; do
				python run_mp.py --model_type tinybert --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization $opt
			done
		done
	done
done

<<c
for opt in "topk" "desired"; do
	for delta_val in 0.1 0.15 0.2 0.35 0.5; do
		for seed_val in 42 84 126 168 210; do
			if [[$opt -eq topk && $delta -eq 0.1 && $seed_val -eq [42 84 126]]]; then
				continue
			fi
			python run.py --model_type tinybert --dataset_name toy-spam --sorting confidence --seed $seed_val --delta $delta_val --optimization $opt
		done
 	done
done
c
# for s in 10; do
# 	for d in results1/tinybert/corona/confidence/seed/$s/* ; do
# 		mkdir --parents results1/tinybert/corona/confidence/$(basename $d)/seed-$s;
# 		for f in $d/*; do
# 			mv $f results1/tinybert/corona/confidence/$(basename $d)/seed-$s/$(basename $f)
# 		done
# 	done
# done