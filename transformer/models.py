import torch
import torch.nn as nn
from torchtext.vocab import GloVe
import math

def create_embedding_layer(input_vocabulary, embedding_dim, statistics):
    embedding_glove = GloVe(name='6B', dim=embedding_dim)
    vocab_size = len(input_vocabulary)
    embedding_dim+=len(statistics)
    weight_matrix = torch.zeros((vocab_size, embedding_dim))
    embedding_glove_itos = embedding_glove.itos
    indices_to_zero = []
    for i, word in enumerate(input_vocabulary.itos):
        if word in embedding_glove_itos:
            if len(statistics)>0:
                weight_matrix[i, :-len(statistics)] = embedding_glove[word]
                for j in range(len(statistics)):
                    weight_matrix[i, -(j+1)] = statistics[j][word]
            else:
                weight_matrix[i] = embedding_glove[word]
            indices_to_zero.append(i)
        else:
            weight_matrix[i] = torch.normal(mean=0.0, std=1.0, size=(1, embedding_dim))
            
    
    #pad token is zeros and unk token is average
    weight_matrix[1] = torch.zeros(size=(1, embedding_dim))
    weight_matrix[0] = torch.mean(weight_matrix, dim=0)
    
    embedding_layer = nn.Embedding.from_pretrained(weight_matrix)
    #embedding_layer.requires_grad_(True)

    return embedding_layer, indices_to_zero


class VanillaGRU(nn.Module):
    def __init__(self, input_vocabulary, embedding_dim, hidden_dim, num_layers, output_dim, dropout, statistics=[], regression = False):
        super().__init__()

        self.embedding_layer, self.indices_to_zero = create_embedding_layer(input_vocabulary, embedding_dim, statistics)
        
        self.GRU_layer = nn.GRU(input_size=embedding_dim+len(statistics), hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout)

        self.dropout_layer = nn.Dropout(dropout)

        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        self.log_softmax = nn.LogSoftmax(dim=1)
        
        self.regression = regression

    def forward(self, x):
        _, batch_size = x.shape
        embedded = self.embedding_layer(x)
        lstm_out, _ = self.GRU_layer(embedded)
        out = self.dropout_layer(lstm_out)
        out = self.fc(out[-1])
        if self.regression:
            return out
        return self.log_softmax(out)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        self.proj_Q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True)
        self.relu_Q = nn.ReLU()

        self.proj_K = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True)
        self.relu_K = nn.ReLU()

        self.proj_V = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True)
        self.relu_V = nn.ReLU()

        self.multi_head_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)

        self.final_linear_layer = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True)
        self.relu_final_liner_layer = nn.ReLU()

    def forward(self, x):
        Q = self.proj_Q(x)
        Q_relu = self.relu_Q(Q)

        K = self.proj_K(x)
        K_relu = self.relu_K(K)

        V = self.proj_V(x)
        V_relu = self.relu_V(V)

        attention_output, attention_weights = self.multi_head_attention(Q_relu, K_relu, V_relu)

        return self.relu_final_liner_layer(self.final_linear_layer(attention_output)), attention_weights

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttentionNet(nn.Module):
    def __init__(self, input_vocabulary, embed_dim, num_heads, dropout, output_classes, two_attention_layers = False):
        super().__init__()
        self.embedding_layer, self.indices_to_zero  = create_embedding_layer(input_vocabulary, embed_dim, [])
        self.positional_embedded = PositionalEncoding(embed_dim, dropout)

        self.two_attention_layers = two_attention_layers
        self.first_attention_layer = MultiHeadAttentionLayer(embed_dim, num_heads, dropout)
        self.first_dropout_layer = nn.Dropout(p=dropout)
        if self.two_attention_layers:
            self.second_attention_layer = MultiHeadAttentionLayer(embed_dim, num_heads, dropout)
            self.second_dropout_layer = nn.Dropout(p=dropout)

        self.after_average_layer = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True)
        self.relu_average = nn.ReLU()

        self.dropout_representations = nn.Dropout(p=dropout)
        self.classification = nn.Linear(in_features=embed_dim, out_features=output_classes, bias=True)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embedded = self.embedding_layer(x)

        embedded = self.positional_embedded(embedded)
        
        attention_output = embedded
        attention_weights = None
        
        attention_output, attention_weights = self.first_attention_layer(attention_output)
        attention_output = self.first_dropout_layer(attention_output)
        
        if self.two_attention_layers:
            attention_output, attention_weights = self.second_attention_layer(attention_output)
            attention_output = self.second_dropout_layer(attention_output)

        averaged = torch.mean(attention_output, dim=0)

        out = self.after_average_layer(averaged)
        out = self.relu_average(out)
        out = self.dropout_representations(out)
        out = self.classification(out)
        return self.log_softmax(out), attention_weights