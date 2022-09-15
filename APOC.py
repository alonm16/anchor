import torch
import matplotlib.pyplot as plt
import numpy as np

model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_scores(sentences):
    softmax = torch.nn.Softmax()
    encoded = [[101] +[tokenizer.vocab[token] for token in tokens] + [102]         
               for tokens in sentences]
    #encoded = tokenizer.encode(sentences, add_special_tokens=True, return_tensors="pt").to(device)
    to_pred = torch.tensor(encoded, device=device)
    outputs = softmax(model(to_pred)[0])
    return outputs.detach().cpu().numpy()

def remove_tokens(removed_tokens, sentences):
    return [['[PAD]' if token in removed_tokens else token for token in sentence] for sentence in sentences]

def apoc_predictions(sentences_arr, labels):
    predictions_arr = []
    for sentences in sentences_arr:
        predictions = [predict_scores([sentence])[0][label] for sentence, label in zip(sentences, labels)]
        predictions_arr.append(predictions)
    return predictions_arr
        
def calc_apoc(predictions_arr):
    predictions_arr = np.array(predictions_arr)
    orig_predictions = np.array(predictions_arr[0])
    N = len(orig_predictions)
    values = []
    
    for k in range(len(predictions_arr)):
        value = sum([sum(orig_predictions - predictions_arr[i])/N for i in range(k+1)])/(k+1)
        values.append(value) 
    return values

def apoc_global(tokens_to_remove, sentences, labels):
    tokenized_sentences = [tokenizer.tokenize(example) for example in sentences]
    removed_sentences_arr = [remove_tokens(tokens_to_remove[:i], tokenized_sentences) for i in range(len(tokens_to_remove)+1)]
    predictions_arr = apoc_predictions(removed_sentences_arr, labels)
    apoc_scores = calc_apoc(predictions_arr)
    plt.plot(range(len(apoc_scores)), apoc_scores)
    plt.xlabel('# of features removed per sentence')
    plt.ylabel('APOC - global')