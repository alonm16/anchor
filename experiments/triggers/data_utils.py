from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer as Tokenizer
import pandas as pd
import pickle

def create_instance(sentence, tokenizer):
    sentence = tokenizer.tokenize(sentence)
    token_indexer = {"tokens": SingleIdTokenIndexer()}
    tokens = TextField(sentence, token_indexer)
    return Instance({'tokens':tokens})


def get_dev_sst():
    
    with open('triggers/dev_examples.pickle', 'rb') as handle:
            dev_data = pickle.load(handle)
    with open('triggers/dev_labels.pickle', 'rb') as handle:
            dev_labels = pickle.load(handle)
    
    dev_examples = []
    for example, label in zip(dev_data, dev_labels):
        example = [word.text.lower() for word in example]
        example = ' '.join(example)
        dev_examples.append(example)
    
    return dev_examples, dev_labels
    