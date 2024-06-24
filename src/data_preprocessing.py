import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import os
import joblib

hashtag_vectorizer = CountVectorizer()
caption_vectorizer = CountVectorizer()

def load_config(config_path=None):
    if config_path is None:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        config_path = os.path.join(base_path, 'project', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def preprocess_data(df, is_prediction=False, fit_vectorizers=False):
    if not is_prediction:
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
    else:
        df['hour'] = df['time_of_day'].astype(int)
        df['caption'] = df['description']

    df['hashtags'] = df['hashtags'].fillna('')
    df['caption'] = df['caption'].fillna('')

    if fit_vectorizers:
        hashtags = hashtag_vectorizer.fit_transform(df['hashtags']).toarray()
        captions = caption_vectorizer.fit_transform(df['caption']).toarray()
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        joblib.dump(hashtag_vectorizer, os.path.join(base_path, 'project', 'results', 'models', 'hashtag_vectorizer.pkl'))
        joblib.dump(caption_vectorizer, os.path.join(base_path, 'project', 'results', 'models', 'caption_vectorizer.pkl'))
    else:
        hashtags = hashtag_vectorizer.transform(df['hashtags']).toarray()
        captions = caption_vectorizer.transform(df['caption']).toarray()

    df = df.drop(columns=['hashtags', 'caption'], errors='ignore')
    df = pd.concat([df, pd.DataFrame(hashtags), pd.DataFrame(captions)], axis=1)

    if not is_prediction:
        scaler = MinMaxScaler()
        df['likes'] = scaler.fit_transform(df[['likes']])

    return df

def prepare_datasets(df, config):
    X = df.drop(columns=['likes'])
    y = df['likes']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=config['data']['test_size'], random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=config['data']['validation_size'], random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    config = load_config()
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(base_path, 'project', config['data']['raw_data_path'])

    if not os.path.exists(data_path):
        print(f"Error: The file {data_path} does not exist.")
        return

    df = pd.read_csv(data_path)
    df = preprocess_data(df, fit_vectorizers=True)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets(df, config)

    processed_data_path = os.path.join(base_path, 'project', config['data']['processed_data_path'])
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    X_train.to_csv(os.path.join(processed_data_path, 'X_train.csv'), index=False)
    X_val.to_csv(os.path.join(processed_data_path, 'X_val.csv'), index=False)
    X_test.to_csv(os.path.join(processed_data_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_path, 'y_train.csv'), index=False)
    y_val.to_csv(os.path.join(processed_data_path, 'y_val.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_path, 'y_test.csv'), index=False)

if __name__ == "__main__":
    main()
