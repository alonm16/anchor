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

def sentiment_preprocessor(text):
    text =re.sub('<[^>]*>', '', text)
    text = re.sub('\s+',' ', text)
    text=text.replace('\n','')
    text=text.replace('(\)','')
    text=text.lower()
    text=text.replace('-',' ')
    text=text.replace('\ ' ,'')
    if text[-1]=='.':
        text = text[:-1]
    
    return text.strip()
    
def twitter_preprocess(text_string):
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
    parsed_text = re.sub('#', '', text_string)
    parsed_text = re.sub(space_pattern, ' ', parsed_text)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
   
    return parsed_text.strip()

def sentiment_ds(path = f'{ds_dir}/sentiment.csv'):
    df = pd.read_csv(path).drop(columns=['Unnamed: 0'])
    df['label'] = df['label'].map({'positive': True, 'negative': False})
    df['text'] = df['review'].apply(sentiment_preprocessor)
    df = df[['text', 'label']]
    return prepare_ds(df)

def offensive_ds(path=f'{ds_dir}/offensive.csv'):
    df = pd.read_csv(path, encoding = 'latin-1').drop(columns=['Unnamed: 0', 'count'])
    df = df[df['class'].isin([1,2])]
    df['label'] = df['class'].map({1: True, 2: False}) 
    df['text'] = df['tweet'].apply(twitter_preprocess)
    df = df[['text', 'label']]
    neg_df = df[df['label']==False]
    pos_df = df[df['label']==True][:len(neg_df)]
    df = pd.concat([pos_df, neg_df])
    return prepare_ds(df)


def corona_ds(path = f'{ds_dir}/corona_train.csv'):
    df = pd.read_csv(path, encoding='latin-1')[['OriginalTweet', 'Sentiment']]
    df = df[df['Sentiment']!='Neutral']
    df['label'] = df['Sentiment'].map({'Positive': True, 'Extremely Positive': True,'Negative': False, 'Extremely Negative': False}) 
    df['text'] = df['OriginalTweet'].apply(twitter_preprocess)
    df = df[:17000][['text', 'label']]
    
    return prepare_ds(df)

def sentiment_twitt_dataset(path = f'{ds_dir}/sentiment_twitter.csv'):    
    df = pd.read_csv(path, encoding ='ISO-8859-1')
    df['label'] = df['target'].map({'POSITIVE': True, 'NEGATIVE': False})
    df['text'] = df['text'].apply(twitter_preprocess)
    df = df[['text', 'label']]
    neg_df = df[df['label']==False][:5000]
    pos_df = df[df['label']==True][:len(neg_df)]
    df = pd.concat([pos_df, neg_df])
    return prepare_ds(df) 
    
def counter_dataset(path = f'{ds_dir}/counter_ds.csv'):  
    df = pd.read_csv(path)
    df['label'] = df['Sentiment'].map({'Positive': True, 'Negative': False})
    df['text'] = df['Text'].apply(sentiment_preprocessor)
    df = df[['text', 'label']]
    return prepare_ds(df)

def dilemma_dataset(path = f'{ds_dir}/TheSocialDilemma.csv'):  
    df = pd.read_csv(path)
    df = df[df['Sentiment']!='Neutral']
    df['label'] = df['Sentiment'].map({'Positive': True, 'Negative': False})
    df['text'] = df['text'].apply(twitter_preprocess)
    df = df[['text', 'label']]
    neg_df = df[df['label']==False][:3573]
    pos_df = df[df['label']==True][:len(neg_df)]
    df = pd.concat([pos_df, neg_df])
    return prepare_ds(df, 0.15)
    
    
def prepare_ds(df, test_size = 0.2):
    set_seed()
    train_df, test_df = train_test_split(df, test_size=test_size)
    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df).remove_columns(['__index_level_0__'])
    ds['test'] = Dataset.from_pandas(test_df).remove_columns(['__index_level_0__'])
    return ds

def get_ds(ds_name):
    ds_dict = {
                "sentiment": sentiment_ds,
                "offensive": offensive_ds,
                "corona": corona_ds,
                "sentiment_twitter": sentiment_twitt_dataset,
                "counter": counter_dataset,
                "dilemma": dilemma_dataset
              }
    return ds_dict[ds_name]()

def preprocess_examples(ds, max_example_len = 90):
    examples = ds['train'].filter(lambda x: 20 < len(x['text']) < max_example_len )
    return examples['text']
    
