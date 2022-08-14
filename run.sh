python run.py --dataset_name sentiment --sorting polarity --optimization topk
python run.py --dataset_name sentiment --sorting confidence --optimization topk
python run.py --dataset_name corona --sorting polarity --optimization topk --examples_max_length 150 
python run.py --dataset_name corona --sorting confidence --optimization topk --examples_max_length 150 
python run.py --dataset_name sentiment --sorting polarity --optimization lossy
python run.py --dataset_name sentiment --sorting confidence --optimization lossy
python run.py --dataset_name corona --sorting polarity --optimization lossy --examples_max_length 150 
python run.py --dataset_name corona --sorting confidence --optimization lossy --examples_max_length 150
python run.py --dataset_name sentiment --sorting polarity --optimization desired
python run.py --dataset_name sentiment --sorting confidence --optimization desired
python run.py --dataset_name corona --sorting polarity --optimization desired --examples_max_length 150 
python run.py --dataset_name corona --sorting confidence --optimization desired --examples_max_length 150 

<<note
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
