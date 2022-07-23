python run.py --dataset_name sentiment --optimization topk --sort_function polarity --examples_max_length 90
python run.py --dataset_name sentiment --optimization topk-confidence --sort_function confidence --examples_max_length 90
python run.py --dataset_name offensive --optimization topk --sort_function polarity --examples_max_length 90
python run.py --dataset_name offensive --optimization topk-confidence --sort_function confidence --examples_max_length 90
python run.py --dataset_name corona --optimization topk --sort_function polarity --examples_max_length 150
python run.py --dataset_name corona --optimization topk-confidence --sort_function confidence --examples_max_length 150
