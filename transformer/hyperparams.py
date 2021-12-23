def hyperparams_for_vanilla():
    hp = dict(
        embedding_dim=100,
        batch_size=32,
        hidden_dim=256,
        num_layers=2,
        dropout=0.7,
        lr = 0.0005,
        early_stopping = 5,
        output_classes = 2
    )
    return hp


def hyperparams_for_attention():
    hp = dict(
        embedding_dim=100,
        batch_size=16,
        dropout=0.6,
        lr = 0.0005,
        two_atten_layers=False,
        num_heads=10,
        early_stopping = 5,
        output_classes = 2
    )
    return hp

#################################
def hyperparams_for_gru_trinary():
    hp = dict(
        embedding_dim=100,
        batch_size=16,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        lr = 0.00005,
        early_stopping = 5,
        output_classes = 3
    )
    return hp


def hyperparams_for_attention_trinary():
    hp = dict(
        embedding_dim=100,
        batch_size=8,
        dropout=0.2,
        lr = 0.00005,
        two_atten_layers=False,
        num_heads=2,
        early_stopping = 5,
        output_classes = 3
    )
    return hp

def hyperparams_for_gru_binary():
    hp = dict(
        embedding_dim=100,
        batch_size=32,
        hidden_dim=256,
        num_layers=2,
        dropout=0.7,
        lr = 0.0005,
        early_stopping = 5,
        output_classes = 2
    )
    return hp


def hyperparams_for_attention_binary():
    hp = dict(
        embedding_dim=100,
        batch_size=16,
        dropout=0.6,
        lr = 0.0005,
        two_atten_layers=False,
        num_heads=10,
        early_stopping = 5,
        output_classes = 2
    )
    return hp

def hyperparams_for_gru_fine_grained():
    hp = dict(
        embedding_dim=100,
        batch_size=16,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        lr = 0.00005,
        early_stopping = 5,
        output_classes = 5
    )
    return hp


def hyperparams_for_attention_fine_grained():
    hp = dict(
        embedding_dim=100,
        batch_size=8,
        dropout=0.2,
        lr = 0.00005,
        two_atten_layers=False,
        num_heads=2,
        early_stopping = 5,
        output_classes = 5
    )
    return hp