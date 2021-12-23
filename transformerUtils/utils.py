import torchtext.data
import torchtext.datasets
import torch
import os
import transformerUtils.hyperparams as hyperparams
import transformerUtils.models as models
from torchtext.legacy import data
import random
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = os.path.expanduser('~/.pytorch-datasets')

type_dataset = 'binary' # What kind of dataset to train on trinary, fine_grained or binary


# Preparing Dataset

def load_dataset(fine_grained = False):
    # torchtext Field objects parse text (e.g. a review) and create a tensor representation

    # This Field object will be used for tokenizing the movie reviews text
    # For this application, tokens ~= words

    review_parser = torchtext.legacy.data.Field(
        sequential=True, use_vocab=True, lower=True, dtype=torch.long,
        tokenize='spacy', tokenizer_language='en_core_web_sm', init_token='sos', eos_token='eos'
    )

    # This Field object converts the text labels into numeric values (0,1,2)
    label_parser = torchtext.legacy.data.Field(
        is_target=True, sequential=False, unk_token=None, use_vocab=True
    )
    
    # Load SST, tokenize the samples and labels
    # ds_X are Dataset objects which will use the parsers to return tensors
    ds_train, ds_valid, ds_test = torchtext.legacy.datasets.SST.splits(
        review_parser, label_parser, root=data_dir, fine_grained=fine_grained
    )


    return review_parser, label_parser, ds_train, ds_valid, ds_test


def build_vocabulary(review_parser, label_parser, ds_train, min_freq):
    review_parser.build_vocab(ds_train, min_freq = min_freq)
    label_parser.build_vocab(ds_train, min_freq= min_freq)

    print(f"Number of tokens in training samples: {len(review_parser.vocab)}")
    print(f"Number of tokens in training labels: {len(label_parser.vocab)}")
    return review_parser, label_parser


def print_dataset(ds, classes, type_ds):
    print(f'Number of {type_ds} samples: {len(ds)}') 
    for class_type in classes:
        number_of_class = len([example for example in ds if example.label == class_type])
        print(f'Number of {class_type} examples {number_of_class}')
    print('------------------------------------------------------')
    

# Create Binary Dataset

def filter_neutral(ds):
    ds.examples = [example for example in ds.examples if example.label != 'neutral']
    

def create_binary_dataset():
    review_parser, label_parser, ds_train, ds_valid, ds_test = load_dataset(fine_grained = False)
    review_parser, label_parser = build_vocabulary(review_parser, label_parser, ds_train, min_freq=5)
    filter_neutral(ds_train)
    filter_neutral(ds_valid)
    filter_neutral(ds_test)
    print_dataset(ds_train, ['positive', 'negative'], 'training')
    print_dataset(ds_valid, ['positive', 'negative'], 'validation')
    print_dataset(ds_test, ['positive', 'negative'], 'test')
    return review_parser, label_parser, ds_train, ds_valid, ds_test


def create_sentiment_dataset(path='datasets/sentiment-sentences'):

    text_field = torchtext.legacy.data.Field(
        sequential=True, use_vocab=True, lower=True, dtype=torch.long,
        tokenize='spacy', tokenizer_language='en_core_web_sm', init_token='sos', eos_token='eos'
    )

    # This Field object converts the text labels into numeric values (0,1,2)
    label_field = torchtext.legacy.data.Field(
        is_target=True, sequential=False, unk_token=None, use_vocab=True
    )

    fields = [('text', text_field), ('label', label_field)]

    examples = []
    f_names = ['rt-polarity.pos', 'rt-polarity.neg']
    f_labels = ['negative', 'positive']
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), 'rb'):
            try:
                line = line.decode('utf8')
            except:
                continue
            examples.append(data.Example.fromlist([line, f_labels[l]], fields))

    random.seed(10)
    random.shuffle(examples)
    train_examples = examples[:7000]
    val_examples = examples[7000:8000]
    test_examples = examples[8000:]
    ds_train = data.Dataset(examples=train_examples, fields=fields)
    ds_val = data.Dataset(examples=val_examples, fields=fields)
    ds_test = data.Dataset(examples=test_examples, fields=fields)
    
    review_parser, label_parser = build_vocabulary(text_field, label_field, ds_train, min_freq=5)
    
    return review_parser, label_parser, ds_train, ds_val, ds_test


def counter_test():
    f = pd.read_csv('counter_ds/test.tsv', sep='\t')
    label_dict = {'Negative': 0, 'Positive': 1}
    f['label'] = [label_dict[label] for label in f['Sentiment']]

    return f['Text'].to_numpy(), f['label'].to_numpy()


def load_counter(path, label_dict, fields):
    df= pd.read_csv(path, sep='\t')

    texts = df['Text']
    labels = [label_dict[label] for label in df['Sentiment']]
    examples= []
    for x, y in zip(texts, labels):
        examples.append(data.Example.fromlist([x, y], fields))
    
    return examples
 
    
def counter_dataset():
    text_field = torchtext.legacy.data.Field(
        sequential=True, use_vocab=True, lower=True, dtype=torch.long,
        tokenize='spacy', tokenizer_language='en_core_web_sm', init_token='sos', eos_token='eos')

    # This Field object converts the text labels into numeric values (0,1,2)
    label_field = torchtext.legacy.data.Field(is_target=True, sequential=False, unk_token=None, use_vocab=True)
    fields = [('text', text_field), ('label', label_field)]
    label_dict = {'Negative': 0, 'Positive': 1}

    train_examples = load_counter('counter_ds/train.tsv', label_dict, fields)
    val_examples = load_counter('counter_ds/dev.tsv', label_dict, fields)
    test_examples = load_counter('counter_ds/train.tsv', label_dict, fields)
    
    ds_train = data.Dataset(examples=train_examples, fields=fields)
    ds_val = data.Dataset(examples=val_examples, fields=fields)
    ds_test = data.Dataset(examples=test_examples, fields=fields)

    review_parser, label_parser = build_vocabulary(text_field, label_field, ds_train, min_freq=5)

    return review_parser, label_parser, ds_train, ds_val, ds_test
       
#  Load Model

def load_hyperparams(model_type, type_dataset):
    hp = None
    if model_type == 'gru':
        hp = hyperparams.hyperparams_for_gru_binary()
    elif model_type == 'attention':
        hp = hyperparams.hyperparams_for_attention_binary()
    print(hp)
    return hp


def load_model(model_name, path, review_parser):
    if model_name == 'gru':
        hp = load_hyperparams(model_name, type_dataset)
        model = models.VanillaGRU(review_parser.vocab, hp['embedding_dim'], hp['hidden_dim'], hp['num_layers'], hp['output_classes'], hp['dropout']).to(device)
    elif model_name == 'attention':
        hp = load_hyperparams(model_name, type_dataset)
        model = models.MultiHeadAttentionNet(input_vocabulary=review_parser.vocab, embed_dim=hp['embedding_dim'], num_heads=hp['num_heads'], 
                                           dropout=hp['dropout'], two_attention_layers=hp['two_atten_layers'], output_classes=hp['output_classes']).to(device)
    saved_state = torch.load(path, map_location=device)
    model.load_state_dict(saved_state)
    model.eval()
    print(model)
    
    return model