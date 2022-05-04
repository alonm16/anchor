import torch
from transformer import hyperparams, models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

type_dataset = 'binary' # What kind of dataset to train on trinary, fine_grained or binary

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
    #somtimes add ["model_state"]
    model.load_state_dict(saved_state['model_state'])
    model.eval()
    print(model)
    
    return model