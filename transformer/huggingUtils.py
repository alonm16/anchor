from pathlib import Path
import random
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score, mean_squared_error

def set_seed(seed=42):
    """
    ensures same results every run
    :param seed: seed for ensuring same results
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_model(model_name = 'roberta-base'):
    """
    loads model from huggingface
    :param model_name: name for model
    :return: loaded model
    """
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

def load_data(path = "clean_data/baby", dev_path = "clean_data/baby"):
    """
    loads train and test set
    :param path: path for data directory
    :return: train and test set
    """
    TRAIN_PATH = Path(path)
    DEV_PATH = Path(dev_path)
    data_files = {
        'train': str(TRAIN_PATH / 'train.csv'),
        'test': str(DEV_PATH / 'dev.csv')
    }
    
    raw_datasets = load_dataset("csv", data_files=data_files).remove_columns(['Unnamed: 0'])
    return raw_datasets

def load_unlabeled(path = "clean_data/baby"):
    """
    loads train and test set
    :param path: path for data directory
    :return: train and test set
    """
    DATA_PATH = Path(path)
    data_files = {
        'unlabeled': str(DATA_PATH / 'unlabeled.csv'),
    }
    
    raw_datasets = load_dataset("csv", data_files=data_files).remove_columns(['Unnamed: 0'])
    return raw_datasets

def tokenize_dataset(raw_datasets, add_label = True, tokenizer_name = 'roberta-base', max_length= 128, truncation= True, padding= "max_length"):
    """
    tokenize dataset according to parameters
    :param raw_datasets: train and test datasets
    :return: tokenized dataset
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_datasets = raw_datasets.map(tokenizer, input_columns='review', remove_columns=["review"],fn_kwargs={"max_length": max_length, "truncation": truncation, "padding": padding})
    tokenized_datasets.set_format('torch')
    
    if add_label:
        for split in tokenized_datasets:
            tokenized_datasets[split] = tokenized_datasets[split].add_column('label', raw_datasets[split]['label'])

    return tokenized_datasets
    
def metric_fn(predictions):
    """
    metric function for model
    :param predictions: predictions of model
    :return: calculated metric
    """
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    return {'accuracy': mean_squared_error(preds, labels)}

def train(model_seq_classification, tokenized_datasets, evaluate=False):
    """
    fine tune huggingface model
    :param model_seq_classification: pre trained model
    :param tokenized_datasets: datasets of tokens
    """
    OUT_PATH = Path("results")
    args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=32, per_device_eval_batch_size=32, save_strategy='no', metric_for_best_model='dev_accuracy', greater_is_better=True, evaluation_strategy='epoch', do_train=True, num_train_epochs=10, report_to='none')
    
    trainer = Trainer(
    model=model_seq_classification,
    args=args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=metric_fn)
    
    if evaluate: 
        return trainer.evaluate()
    else:
        return trainer.train()
        
def evaluate(model, tokenizer_name = 'roberta-base', data_path='clean_data/office_products/dev.csv'):
    data_files = {
        'test': data_path
    }
    
    raw_datasets = load_dataset("csv", data_files=data_files).remove_columns(['Unnamed: 0'])
    
    tokenized_data = tokenize_dataset(raw_datasets, tokenizer_name=tokenizer_name)
    
    tokenized_data['train'] = None
    
    if type(model) == str:
        model = load_model(model_name)
    
    return train(model, tokenized_data, evaluate=True)
    
