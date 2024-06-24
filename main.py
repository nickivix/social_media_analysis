import os
import pandas as pd
from src.data_preprocessing import preprocess_data, prepare_input
from src.train import build_model, train_model
from src.instagram_data_collector import collect_data_from_file

def main():
    data_path = 'data/raw/instagram_posts.csv'
    df = pd.read_csv(data_path, parse_dates=['date'])
    df = preprocess_data(df)
    X, y = prepare_input(df)

    # Split the data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, test_size=0.3, validation_size=0.15)

    model = build_model(X_train.shape)
    train_model(model, X_train, y_train, X_val, y_val, save_path='results/models/model.h5')

if __name__ == "__main__":
    main()
