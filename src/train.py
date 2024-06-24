import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
import pandas as pd
import os
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def build_model(input_shape, config):
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128, input_length=input_shape))
    for layer in config['model']['gru_layers']:
        model.add(GRU(units=layer['units'], activation=layer['activation'], return_sequences=layer['return_sequences']))
        model.add(Dropout(config['model']['dropout_rate']))
    model.add(Dense(2, activation=config['model']['output_activation']))  # Output layer with 2 units

    optimizer = tf.keras.optimizers.Adam(learning_rate=config['model']['learning_rate'])
    model.compile(optimizer=optimizer, loss=config['model']['loss'], metrics=config['model']['metrics'])

    return model

def train_model(model, X_train, y_train, X_val, y_val, config):
    model.fit(X_train, y_train, epochs=config['training']['epochs'], batch_size=config['training']['batch_size'], validation_data=(X_val, y_val))
    model.save(config['training']['save_model_path'])

if __name__ == "__main__":
    config = load_config()

    processed_data_path = config['data']['processed_data_path']
    X_train = pd.read_csv(os.path.join(processed_data_path, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(processed_data_path, 'X_val.csv'))
    y_train = pd.read_csv(os.path.join(processed_data_path, 'y_train.csv'))
    y_val = pd.read_csv(os.path.join(processed_data_path, 'y_val.csv'))

    input_shape = X_train.shape[1]
    model = build_model(input_shape, config)
    train_model(model, X_train, y_train, X_val, y_val, config)
