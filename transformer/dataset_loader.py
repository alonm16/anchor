import torchtext.data
import torchtext.datasets
import torch
import os
import pathlib
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import random
import pandas as pd
import re
from torchtext.legacy import data
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = os.path.expanduser('~/.pytorch-datasets')


# Preparing Dataset

def load_sst(fine_grained = False):
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


def create_sentiment_dataset(path='dataset/sentiment-sentences'):

    text_field = torchtext.legacy.data.Field(
        sequential=True, use_vocab=True, lower=True, dtype=torch.long,
        tokenize='spacy', tokenizer_language='en_core_web_sm', init_token='sos', eos_token='eos'
    )

    # This Field object converts the text labels into numeric values (0,1,2)
    label_field = torchtext.legacy.data.Field(
        is_target=True, sequential=False, unk_token=None, use_vocab=False, dtype=torch.long
    )

    fields = [('text', text_field), ('label', label_field)]

    main_dir_path= pathlib.Path(__file__).parent.parent.resolve().as_posix()
    path = '/'.join([main_dir_path, path])

    examples = []
    f_names = ['rt-polarity.pos', 'rt-polarity.neg']
    f_labels = ['positive', 'negative']
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

def not_finished_create_sentiment_dataset(path='dataset/sentiment-sentences'):
    def collate_batch(batch):
        label_list, text_list = [], []
        for (_text, _label) in batch:
             label_list.append(_label)
             processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
             text_list.append(processed_text)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.stack(text_list)
        return  text_list.to(device), label_list.to(device)

    tokenizer = get_tokenizer('basic_english')
    
    def yield_tokens(data_iter):
        for text, _  in data_iter:
            yield tokenizer(text)
    
    main_dir_path= pathlib.Path(__file__).parent.parent.resolve().as_posix()
    path = '/'.join([main_dir_path, path])
    X, y = [], []
    f_names = ['rt-polarity.pos', 'rt-polarity.neg']
    f_labels = [1, 0]
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), 'rb'):
            try:
                line = line.decode('utf8')
            except:
                continue
            X.append(line)
            y.append(f_labels[l])
    
    train_iter = iter(list(zip(X,y)))
    train_dataset = to_map_style_dataset(train_iter)
    vocab = build_vocab_from_iterator(yield_tokens(train_dataset), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    
    text_pipeline = lambda x: vocab(tokenizer(x))
    
    random.seed(10)
    #random.shuffle(examples)

    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
    
    return dataloader


def counter_test():
    main_dir_path= pathlib.Path(__file__).parent.parent.resolve().as_posix()
    path = '/'.join([main_dir_path, 'dataset/counter_ds/test.tsv'])
    f = pd.read_csv(path, sep='\t')
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

def spam_dataset(path='dataset/spam.csv'):

    text_field = torchtext.legacy.data.Field(
        sequential=True, use_vocab=True, lower=True, dtype=torch.long,
        tokenize='spacy', tokenizer_language='en_core_web_sm', init_token='sos', eos_token='eos'
    )

    # This Field object converts the text labels into numeric values (0,1,2)
    label_field = torchtext.legacy.data.Field(
        is_target=True, sequential=False, unk_token=None, use_vocab=True
    )

    fields = [('text', text_field), ('label', label_field)]
    
    df = pd.read_csv(path, encoding = 'latin-1')
    df.drop(df.columns[[2,3,4]], axis = 1, inplace = True)
    df.columns = ['target','messages']
    
    spams =[]
    hams = []

    for index, row in df.iterrows():
        if row['target'] == 'spam':
            target = 'positive'
            spams.append(data.Example.fromlist([row['messages'], target], fields))
        else:
            target = 'negative'
            hams.append(data.Example.fromlist([row['messages'], target], fields))

    random.seed(10)
    random.shuffle(spams)
    random.shuffle(hams)
    print(len(spams))
    print(len(hams))
    
    train_examples = spams[:500]
    train_examples.extend(hams[:500])
    random.shuffle(train_examples)
    val_examples = spams[500:]
    val_examples.extend(hams[500:750])
    random.shuffle(val_examples)
    ds_train = data.Dataset(examples=train_examples, fields=fields)
    ds_val = data.Dataset(examples=val_examples, fields=fields)
    
    import copy
    ds_for_parser = copy.deepcopy(train_examples)
    ds_for_parser.extend(hams[750:])
    ds_for_parser = data.Dataset(examples=ds_for_parser, fields=fields)
    review_parser, label_parser = build_vocabulary(text_field, label_field, ds_for_parser, min_freq=3)
    
    return review_parser, label_parser, ds_train, ds_val

def offensive_preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text

def offensive_dataset(path='dataset/offensive.csv'):

    text_field = torchtext.legacy.data.Field(
        sequential=True, use_vocab=True, lower=True, dtype=torch.long,
        tokenize='spacy', tokenizer_language='en_core_web_sm', init_token='sos', eos_token='eos'
    )

    # This Field object converts the text labels into numeric values (0,1,2)
    label_field = torchtext.legacy.data.Field(
        is_target=True, sequential=False, unk_token=None, use_vocab=True
    )

    fields = [('text', text_field), ('label', label_field)]
    
    df = pd.read_csv(path).drop(columns=['Unnamed: 0', 'count'])
    
    offensive =[]
    not_offensive = []

    for index, row in df.iterrows():
        tweet = offensive_preprocess(row['tweet'])
        if row['class'] == 1:
            target = 'positive'
            offensive.append(data.Example.fromlist([tweet, target], fields))
        elif row['class'] == 2:
            target = 'negative'
            not_offensive.append(data.Example.fromlist([tweet, target], fields))

    random.seed(10)
    random.shuffle(offensive)
    random.shuffle(not_offensive)
    print(len(offensive))
    print(len(not_offensive))
    
    train_examples = offensive[:3000]
    train_examples.extend(not_offensive[:3000])
    random.shuffle(train_examples)
    val_examples = offensive[3000:4163]
    val_examples.extend(not_offensive[3000:])
    random.shuffle(val_examples)
    ds_train = data.Dataset(examples=train_examples, fields=fields)
    ds_val = data.Dataset(examples=val_examples, fields=fields)
    
    import copy
    ds_for_parser = copy.deepcopy(train_examples)
    ds_for_parser.extend(offensive[4163:])
    ds_for_parser = data.Dataset(examples=ds_for_parser, fields=fields)
    review_parser, label_parser = build_vocabulary(text_field, label_field, ds_for_parser, min_freq=3)
    
    return review_parser, label_parser, ds_train, ds_val