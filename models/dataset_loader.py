import re
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset, DatasetDict

ds_dir = '/home/almr16/anchor/dataset/'

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
    
def preprocess(text):
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
    text = re.sub('#', '', text)
    text = re.sub(space_pattern, ' ', text)
    text = re.sub(giant_url_regex, '', text)
    text = re.sub(mention_regex, '', text)
    text = re.sub('<[^>]*>', '', text)
    text=text.replace('\n','')
    text=text.replace('(\)','')
    text=text.replace('-',' ')
    text=text.replace('\ ' ,'')
    text=text.lower()
    
    if text[-1]=='.':
        text = text[:-1]
   
    return text.strip()

def sentiment_ds(path = f'{ds_dir}/sentiment.csv'):
    df = pd.read_csv(path).drop(columns=['Unnamed: 0'])
    df['label'] = df['label'].map({'positive': True, 'negative': False})
    df['text'] = df['review'].apply(preprocess)
    df = df[['text', 'label']]
    return prepare_ds(df)

def offensive_ds(path=f'{ds_dir}/offensive.csv'):
    df = pd.read_csv(path, encoding = 'latin-1').drop(columns=['Unnamed: 0', 'count'])
    df = df[df['class'].isin([1,2])]
    df['label'] = df['class'].map({1: True, 2: False}) 
    df['text'] = df['tweet'].apply(preprocess)
    df = df[['text', 'label']]
    neg_df = df[df['label']==False]
    pos_df = df[df['label']==True][:len(neg_df)]
    df = pd.concat([pos_df, neg_df])
    return prepare_ds(df)


def corona_ds(path = f'{ds_dir}/corona_train.csv'):
    df = pd.read_csv(path, encoding='latin-1')[['OriginalTweet', 'Sentiment']]
    df = df[df['Sentiment']!='Neutral']
    df['label'] = df['Sentiment'].map({'Positive': True, 'Extremely Positive': True,'Negative': False, 'Extremely Negative': False}) 
    df['text'] = df['OriginalTweet'].apply(preprocess)
    df = df[['text', 'label']]
    
    return prepare_ds(df)

def sentiment_twitt_dataset(path = f'{ds_dir}/sentiment_twitter.csv'):    
    df = pd.read_csv(path, encoding ='ISO-8859-1')
    df['label'] = df['target'].map({'POSITIVE': True, 'NEGATIVE': False})
    df['text'] = df['text'].apply(preprocess)
    df = df[['text', 'label']]
    neg_df = df[df['label']==False][:5000]
    pos_df = df[df['label']==True][:len(neg_df)]
    df = pd.concat([pos_df, neg_df])
    return prepare_ds(df) 
    
def counter_dataset(path = f'{ds_dir}/counter_ds.csv'):  
    df = pd.read_csv(path)
    df['label'] = df['Sentiment'].map({'Positive': True, 'Negative': False})
    df['text'] = df['Text'].apply(preprocess)
    df = df[['text', 'label']]
    return prepare_ds(df)

def dilemma_dataset(path = f'{ds_dir}/TheSocialDilemma.csv'):  
    df = pd.read_csv(path)
    df = df[df['Sentiment']!='Neutral']
    df['label'] = df['Sentiment'].map({'Positive': True, 'Negative': False})
    df['text'] = df['text'].apply(preprocess)
    df = df[['text', 'label']]
    neg_df = df[df['label']==False]
    pos_df = df[df['label']==True][:len(neg_df)]
    df = pd.concat([pos_df, neg_df])
    return prepare_ds(df, val_size = 0.55)

def toy_spam_dataset(path = f'{ds_dir}/toys_and_games.csv'):  
    df = pd.read_csv(path)
    df['text'] = df['text'].apply(preprocess)
    df = df[['text', 'label']]
    # too many sentences
    neg_df = df[df['label']==False][:15000]
    pos_df = df[df['label']==True][:len(neg_df)]
    df = pd.concat([pos_df, neg_df])
    return prepare_ds(df)

def sports_spam_dataset(path = f'{ds_dir}/sports_and_outdoors.csv'):  
    df = pd.read_csv(path)
    df['text'] = df['text'].apply(preprocess)
    df = df[['text', 'label']]
    # too many sentences
    neg_df = df[df['label']==False][:15000]
    pos_df = df[df['label']==True][:len(neg_df)]
    df = pd.concat([pos_df, neg_df])
    return prepare_ds(df)

def home_dataset(path = f'{ds_dir}/Home_and_Kitchen.csv'):  
    df = pd.read_csv(path)
    df['text'] = df['text'].apply(preprocess)
    df = df[['text', 'label']]
    # too many sentences
    neg_df = df[df['label']==False][:12000]
    pos_df = df[df['label']==True][:len(neg_df)]
    df = pd.concat([pos_df, neg_df])
    return prepare_ds(df)
    
    
def prepare_ds(df, val_size = 0.5, test_size = 0.75):
    set_seed()
    train_df, val_df = train_test_split(df, test_size=val_size)
    val_df, test_df = train_test_split(val_df, test_size=test_size)
    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df).remove_columns(['__index_level_0__'])
    ds['val'] = Dataset.from_pandas(val_df).remove_columns(['__index_level_0__'])
    ds['test'] = Dataset.from_pandas(test_df).remove_columns(['__index_level_0__'])
    return ds

def get_ds(ds_name):
    ds_dict = {
                "sentiment": sentiment_ds,
                "offensive": offensive_ds,
                "corona": corona_ds,
                "sentiment_twitter": sentiment_twitt_dataset,
                "counter": counter_dataset,
                "dilemma": dilemma_dataset,
                "toy-spam": toy_spam_dataset,
                "sport-spam": sports_spam_dataset,
                "home-spam": home_dataset
              }
    return ds_dict[ds_name]()

def preprocess_examples(ds, max_example_len = 150, for_retrain = False):
    examples = ds['test'] if not for_retrain else ds['train']
    examples = examples.filter(lambda x: 20 < len(x['text']) < max_example_len)
    return examples['text'], examples['label']
    
