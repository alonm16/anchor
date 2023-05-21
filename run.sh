#!/bin/bash
deltas=(0.1 0.15 0.2 0.35 0.5)
seeds=(42 84 126 168 210)
m_type="deberta"

for ds in "corona" "home-spam" "dilemma"; do
    for seed_val in ${seeds[@]}; do
        for delta_val in ${deltas[@]}; do
            python run_mp.py --model_type $m_type --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val
            for opt in "stop-words" "topk" "desired" "masking"; do
				python run_mp.py --model_type $m_type --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization $opt
            done
            python run_mp.py --model_type $m_type --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization desired masking
            python run_mp.py --model_type $m_type --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization stop-words masking
            python run_mp.py --model_type $m_type --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization stop-words topk
            python run_mp.py --model_type $m_type --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization stop-words desired
            python run_mp.py --model_type $m_type --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization stop-words topk masking
            python run_mp.py --model_type $m_type --dataset_name $ds --sorting confidence --seed $seed_val --delta $delta_val --optimization stop-words topk desired masking	
        done
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
