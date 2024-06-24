import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from data_preprocessing import preprocess_data, load_config

def evaluate_model():
    config = load_config()
    df = pd.read_csv(config['data']['raw_data_path'])
    _, _, X_test, _, _, y_test = preprocess_data(df, config)

    model = tf.keras.models.load_model(config['training']['save_model_path'])
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    evaluate_model()
