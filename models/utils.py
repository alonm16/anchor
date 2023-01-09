from pathlib import Path
import random
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, load_metric, DatasetDict
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class MyGRU(nn.Module):
    def __init__(self, model_name, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.embedding = load_model(model_name).get_input_embeddings()
        self.GRU_layer = nn.GRU(input_size=self.embedding.embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels = None):
        input_ids = torch.transpose(input_ids, 0, 1)
        embedded = self.embedding(input_ids)
        out, _ = self.GRU_layer(embedded)
        out = self.dropout(out)
        out = self.fc(out[-1])
        
        if labels is not None: 
            return (self.criterion(out, labels), out)
        
        return (out,)
    
class MySVM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.vectorizer = None
        
    def train(self, ds, tokenizer):
        train_labels = np.array(ds['train']['label'])
        train = [' '.join(map(str, tokenizer.encode(x))) for x in ds['train']['text']]
        val_labels = np.array(ds['val']['label'])
        val = [' '.join(map(str, tokenizer.encode(x))) for x in ds['val']['text']]
        
        self.vectorizer = CountVectorizer(min_df=1)
        self.vectorizer.fit(train)
        train_vectors = self.vectorizer.transform(train)
        val_vectors = self.vectorizer.transform(val)
        
        model = SVC(probability = True)
        model.fit(train_vectors, train_labels)
        val_pred = model.predict(val_vectors)
        train_pred = model.predict(train_vectors)
        print('Train accuracy', accuracy_score(train_labels, train_pred))
        print('Validation accuracy', accuracy_score(val_labels, val_pred))
        
        self.model = model
        
    def eval(self):
        return self

    def forward(self, input_ids):
        str_input = [' '.join(map(str, cur_input_ids)) for cur_input_ids in input_ids]
        str_input = self.vectorizer.transform(str_input)
        out = self.model.predict_proba(str_input)
        out = torch.from_numpy(out)
        return (out,)
    
class MyLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.vectorizer = None
        
    def train(self, ds, tokenizer):
        train_labels = np.array(ds['train']['label'])
        train = [' '.join(map(str, tokenizer.encode(x))) for x in ds['train']['text']]
        val_labels = np.array(ds['val']['label'])
        val = [' '.join(map(str, tokenizer.encode(x))) for x in ds['val']['text']]
        
        self.vectorizer = CountVectorizer(min_df=1)
        self.vectorizer.fit(train)
        train_vectors = self.vectorizer.transform(train)
        val_vectors = self.vectorizer.transform(val)
        
        model = LogisticRegression()
        model.fit(train_vectors, train_labels)
        val_pred = model.predict(val_vectors)
        train_pred = model.predict(train_vectors)
        print('Train accuracy', accuracy_score(train_labels, train_pred))
        print('Validation accuracy', accuracy_score(val_labels, val_pred))
        
        self.model = model
        
    def eval(self):
        return self

    def forward(self, input_ids):
        str_input = [' '.join(map(str, cur_input_ids)) for cur_input_ids in input_ids]
        str_input = self.vectorizer.transform(str_input)
        out = self.model.predict_proba(str_input)
        out = torch.from_numpy(out)
        return (out,)

def set_seed(seed=42):
    """
    ensures same results every run
    :param seed: seed for ensuring same results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def predicting_sentences(model, tokenizer, device, sentences):
    encoded = [[101] +[tokenizer._convert_token_to_id_with_added_voc(token) for token in tokens] + [102] for tokens in sentences]
    to_pred = torch.tensor(encoded, device=device)
    outputs = model(to_pred)[0]
    print(outputs)
    return torch.argmax(outputs, dim=1).cpu().numpy()
    
def load_model(model_name = 'huawei-noah/TinyBERT_General_4L_312D'):
    """
    loads model from huggingface
    :param model_name: name for model
    :return: loaded model
    """
    if 'svm' in model_name or 'logistic' in model_name:
        return load_sklearn(model_name)
    if 'traced' in model_name:
         return load_traced(model_name)
    if 'gru' in model_name:
        return load_gru(model_name)

    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, torchscript = True)

def load_traced(path):
    return torch.jit.load(path)

def load_sklearn(path):
    from joblib import load
    return load(path) 

def load_gru(path):
    model_name = 'huawei-noah/TinyBERT_General_4L_312D'
    model = MyGRU(model_name, hidden_dim=256, num_layers=2, output_dim=2, dropout=0.0)
    saved_state = torch.load(f'{path}/pytorch_model.bin')
    model.load_state_dict(saved_state)
    return model

def load_data(train_path, dev_path=None):
    """
    loads train and val set
    :param path: path for data directory
    :return: train and val set
    """
    TRAIN_PATH = Path(train_path)
    
    data_files = None
    
    if dev_path is None:
        DEV_PATH = Path(dev_path)    
        data_files = {'train': str(TRAIN_PATH / 'train.csv')}
    else:
        data_files = {
            'train': str(TRAIN_PATH / 'train.csv'),
            'val': str(DEV_PATH / 'dev.csv')
        }
    
    raw_datasets = load_dataset("csv", data_files=data_files).remove_columns(['Unnamed: 0'])
    return raw_datasets


def tokenize_dataset(raw_datasets, tokenizer_name = 'huawei-noah/TinyBERT_General_4L_312D', max_length= 128, truncation= True, padding= "max_length"):
    """
    tokenize dataset according to parameters
    :param raw_datasets: train and val datasets
    :return: tokenized dataset
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_datasets = raw_datasets.map(tokenizer, input_columns='text', remove_columns=["text"],fn_kwargs={"max_length": max_length, "truncation": truncation, "padding": padding})
    tokenized_datasets.set_format('torch')
    
    return tokenized_datasets
    
def metric_fn(predictions):
    """
    metric function for model
    :param predictions: predictions of model
    :return: calculated metric
    """
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    return {'accuracy': accuracy_score(preds, labels)}

def train(model_seq_classification, tokenized_datasets, path, evaluate=False, num_train_epochs=2):
    """
    fine tune huggingface model
    :param model_seq_classification: pre trained model
    :param tokenized_datasets: datasets of tokens
    """
    OUT_PATH = Path(path)
    args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=32, per_device_eval_batch_size=32, save_strategy='epoch', metric_for_best_model='accuracy', load_best_model_at_end=True, greater_is_better=True, evaluation_strategy='epoch', do_train=True, num_train_epochs=num_train_epochs, report_to='none')
    
    trainer = Trainer(
    model=model_seq_classification,
    args=args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    compute_metrics=metric_fn)
    
    if evaluate: 
        return trainer.evaluate()
    else:
        return trainer.train()
        
def per_class_accuracy(folder_name, model_name, dataset_name):
    def eval_filtered(filter_label):
        filtered = ds['val'].filter(lambda x: x['label'] == filter_label)
        filtered_ds = DatasetDict()
        filtered_ds['train'] = filtered
        filtered_ds['val'] = filtered
        filtered_data = tokenize_dataset(filtered_ds, tokenizer_name = model_name, max_length = 64)
        return train(model, filtered_data, path=f'', evaluate = True)['eval_accuracy']
            
    from dataset_loader import get_ds
    ds = get_ds(dataset_name)
    model = AutoModelForSequenceClassification.from_pretrained(f'{folder_name}/{dataset_name}/model', num_labels=2, torchscript=True).eval()
    
    print(f"positive accuracy {eval_filtered(1)}")
    print(f"negative_accuracy {eval_filtered(0)}")