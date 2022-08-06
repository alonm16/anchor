import sys
import os.path
from sklearn.neighbors import KDTree
import torch
import torch.optim as optim
from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.fields import TextField, LabelField
from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
import pickle
import triggers.utils

#import myUtils

# Simple LSTM classifier that uses the final hidden state to classify Sentiment. Based on AllenNLP
class LstmClassifier(Model):
    def __init__(self, word_embeddings, encoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        output = logits
        return output

    
def get_sst():
    # load the binary SST dataset.
    train_path = 'triggers/data/train'
    dev_path = 'triggers/data/dev'
    
    if os.path.isfile(train_path) and os.path.isfile(dev_path):
        with open(train_path, 'rb') as handle:
            train_data = pickle.load(handle)
            
        with open(dev_path, 'rb') as handle:
            dev_data = pickle.load(handle)
    else:
        single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
        # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
        reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                        token_indexers={"tokens": single_id_indexer},
                                                        use_subtrees=True)
        train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
        train_data = list(train_data)
        
        reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                        token_indexers={"tokens": single_id_indexer})
        dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
        dev_data = list(dev_data)
        
        with open(train_path, 'wb') as handle:
            pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(dev_path, 'wb') as handle:
            pickle.dump(dev_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        #test_dataset = reader.read('data/sst/test.txt')
    
    return train_data, dev_data, None

#EMBEDDING_TYPE - what type of word embeddings to use
EMBEDDING_TYPE = "w2v"
def get_embeddings(vocab, word_embedding_dim = 300):
    # Randomly initialize vectors
    if EMBEDDING_TYPE == "None":
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=word_embedding_dim)
       

    # Load word2vec vectors
    elif EMBEDDING_TYPE == "w2v":
        embedding_path = 'triggers/token_embedding'
        if os.path.isfile(embedding_path):
             with open(embedding_path, 'rb') as handle:
                token_embedding = pickle.load(handle)
        else: 
            weight = _read_pretrained_embeddings_file("https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
                                                      embedding_dim=word_embedding_dim,
                                                      vocab=vocab,
                                                      namespace="tokens")
            token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                        embedding_dim=word_embedding_dim,
                                        weight=weight,
                                        trainable=False)
            
            with open(embedding_path, 'wb') as handle:
                pickle.dump(token_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    
    return word_embeddings
    
def load_model_helper(vocab, word_embeddings, train_data=None, dev_data=None):
       # Initialize model, cuda(), and optimizer
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embeddings.get_output_dim(),
                                                  hidden_size=512,
                                                  num_layers=2,
                                                  batch_first=True))
    model = LstmClassifier(word_embeddings, encoder, vocab)
    model.cuda()

    # where to save the model
    model_path = "triggers/models/" + EMBEDDING_TYPE + "_" + "model.th"
    vocab_path = "triggers/models/" + EMBEDDING_TYPE + "_" + "vocab"
    # if the model already exists (its been trained), load the pre-trained weights and vocabulary
    if os.path.isfile(model_path):
        vocab = Vocabulary.from_files(vocab_path)
        model = LstmClassifier(word_embeddings, encoder, vocab)
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))
    # otherwise train model from scratch and save its weights
    else:
        iterator = SimpleDataLoader(dev_dataset, batch_size = 32)
        iterator.index_with(vocab)
        optimizer = optim.Adam(model.parameters())
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_data,
                          validation_dataset=dev_data,
                          num_epochs=5,
                          patience=1,
                          cuda_device=0)
        trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)
        
    model.eval().cuda() 
        
    return model


def load_model():

    train_data, dev_data, _ = get_sst()
    vocab = Vocabulary.from_instances(train_data)
    word_embeddings = get_embeddings(vocab)
    
    model = load_model_helper(vocab, word_embeddings, train_data, dev_data)

    return model, vocab
