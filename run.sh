python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.1
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.15
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.2
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.35
python run.py --dataset_name corona --sorting confidence --examples_max_length 150 --delta 0.5
<<com
python run.py --dataset_name sentiment --sorting confidence --delta 0.1
python run.py --dataset_name sentiment --sorting confidence --delta 0.15
python run.py --dataset_name sentiment --sorting confidence --delta 0.2
python run.py --dataset_name sentiment --sorting confidence --delta 0.35
python run.py --dataset_name sentiment --sorting confidence --delta 0.5
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
com

