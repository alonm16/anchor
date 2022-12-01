python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.1
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.15
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.2
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.35
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.5

python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --optimization lossy
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --optimization topk
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --optimization desired


python run.py --dataset_name dilemma --sorting confidence --examples_max_length 150 --delta 0.1
python run.py --dataset_name dilemma --sorting confidence --examples_max_length 150 --delta 0.15
python run.py --dataset_name dilemma --sorting confidence --examples_max_length 150 --delta 0.2
python run.py --dataset_name dilemma --sorting confidence --examples_max_length 150 --delta 0.35
python run.py --dataset_name dilemma --sorting confidence --examples_max_length 150 --delta 0.5

python run.py --dataset_name dilemma --sorting confidence --examples_max_length 150 --optimization lossy
python run.py --dataset_name dilemma --sorting confidence --examples_max_length 150 --optimization topk
python run.py --dataset_name dilemma --sorting confidence --examples_max_length 150 --optimization desired


python run.py --dataset_name sentiment --sorting confidence --delta 0.1
python run.py --dataset_name sentiment --sorting confidence --delta 0.15
python run.py --dataset_name sentiment --sorting confidence --delta 0.2
python run.py --dataset_name sentiment --sorting confidence --delta 0.35
python run.py --dataset_name sentiment --sorting confidence --delta 0.5

python run.py --dataset_name sentiment --sorting confidence --optimization lossy
python run.py --dataset_name sentiment --sorting confidence --optimization topk
python run.py --dataset_name sentiment --sorting confidence --optimization desired


<<note
change back to not shuffle and return dir name!!!!!!!!!!!!
python run_seed.py --dataset_name corona --sorting confidence --examples_max_length 150 --seed 42
python run_seed.py --dataset_name corona --sorting confidence --examples_max_length 150 --seed 84
python run_seed.py --dataset_name corona --sorting confidence --examples_max_length 150 --seed 126
python run_seed.py --dataset_name corona --sorting confidence --examples_max_length 150 --seed 168
python run.py --dataset_name sentiment --sorting polarity --delta 0.25
python run.py --dataset_name sentiment --sorting polarity --delta 0.30
python run.py --dataset_name sentiment --sorting confidence --delta 0.25
python run.py --dataset_name sentiment --sorting confidence --delta 0.30
python run.py --dataset_name corona --sorting polarity --examples_max_length 150 --delta 0.25
python run.py --dataset_name corona --sorting polarity --examples_max_length 150 --delta 0.30
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.25
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.30
python run_seed.py --dataset_name sentiment --sorting confidence --seed 42
python run_seed.py --dataset_name sentiment --sorting confidence --seed 84
python run_seed.py --dataset_name sentiment --sorting confidence --seed 126
python run_seed.py --dataset_name sentiment --sorting confidence --seed 168
python run.py --dataset_name sentiment --sorting polarity --delta 0.1
python run.py --dataset_name sentiment --sorting polarity --delta 0.15
python run.py --dataset_name sentiment --sorting polarity --delta 0.2
python run.py --dataset_name sentiment --sorting polarity --delta 0.35
python run.py --dataset_name sentiment --sorting polarity --delta 0.5
python run.py --dataset_name corona --sorting polarity --examples_max_length 150 --delta 0.1
python run.py --dataset_name corona --sorting polarity --examples_max_length 150 --delta 0.15
python run.py --dataset_name corona --sorting polarity --examples_max_length 150 --delta 0.2
python run.py --dataset_name corona --sorting polarity --examples_max_length 150 --delta 0.35
python run.py --dataset_name corona --sorting polarity --examples_max_length 150 --delta 0.5
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.1
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.15
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.2
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.35
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.5
python run.py --dataset_name sentiment --sorting confidence --delta 0.1
python run.py --dataset_name sentiment --sorting confidence --delta 0.15
python run.py --dataset_name sentiment --sorting confidence --delta 0.2
python run.py --dataset_name sentiment --sorting confidence --delta 0.35
python run.py --dataset_name sentiment --sorting confidence --delta 0.5
note
