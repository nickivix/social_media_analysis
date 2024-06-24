from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import sys
import http.client
import json
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import predict_with_model
from data_preprocessing import preprocess_data, load_config, main as run_preprocessing
from instagram_data_collector import collect_data_from_file

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'data', 'raw')
DATA_DIR = app.config['UPLOAD_FOLDER']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/existing_data', methods=['GET', 'POST'])
def existing_data():
    if request.method == 'POST':
        selected_metrics = request.form.getlist('metrics')
        if not selected_metrics:
            selected_metrics = ['likes']
        df = pd.read_csv(os.path.join(DATA_DIR, 'instagram_posts.csv'), parse_dates=['date'])
        plot_path, hashtags, words = generate_histogram(df, selected_metrics)
        return render_template('existing_data.html', plot_path=plot_path, title="Існуючі дані", selected_metrics=selected_metrics, hashtags=hashtags, words=words)
    return render_template('existing_data.html', plot_path=None, title="Існуючі дані", selected_metrics=['likes'], hashtags=[], words=[])

@app.route('/search_data', methods=['POST'])
def search_data():
    hashtag = request.form.get('hashtag')
    profiles = get_instagram_profiles(hashtag)
    file_path = os.path.join(DATA_DIR, 'searched_profiles.txt')
    with open(file_path, 'w') as file:
        for profile in profiles:
            file.write(profile + '\n')
    collect_data_from_file(file_path)
    run_preprocessing()
    df = pd.read_csv(os.path.join(DATA_DIR, 'instagram_posts.csv'), parse_dates=['date'])
    selected_metrics = request.form.getlist('metrics')
    if not selected_metrics:
        selected_metrics = ['likes']
    plot_path, hashtags, words = generate_histogram(df, selected_metrics)
    return render_template('existing_data.html', plot_path=plot_path, title="Результати пошуку", selected_metrics=selected_metrics, hashtags=hashtags, words=words)

@app.route('/upload_data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        collect_data_from_file(file_path)
        run_preprocessing()
        df = pd.read_csv(os.path.join(DATA_DIR, 'instagram_posts.csv'), parse_dates=['date'])
        selected_metrics = request.form.getlist('metrics')
        if not selected_metrics:
            selected_metrics = ['likes']
        plot_path, hashtags, words = generate_histogram(df, selected_metrics)
        return render_template('existing_data.html', plot_path=plot_path, title="Завантажені дані", selected_metrics=selected_metrics, hashtags=hashtags, words=words)

@app.route('/predict_popularity', methods=['POST'])
def predict_popularity():
    user_input = {
        'time_of_day': request.form.get('time_of_day'),
        'hashtags': request.form.get('hashtags'),
        'description': request.form.get('description'),
        'last_5_likes': request.form.get('last_5_likes')
    }
    config = load_config()
    model_path = os.path.join(BASE_DIR, 'results', 'models', 'model.h5')

    input_data = pd.DataFrame([user_input])

    global hashtag_vectorizer
    global caption_vectorizer
    hashtag_vectorizer = joblib.load(os.path.join(BASE_DIR, 'results', 'models', 'hashtag_vectorizer.pkl'))
    caption_vectorizer = joblib.load(os.path.join(BASE_DIR, 'results', 'models', 'caption_vectorizer.pkl'))

    input_data = preprocess_data(input_data, is_prediction=True)

    input_data = input_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    input_data_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

    print("Input Data Shape:", input_data_tensor.shape)

    predictions = predict_with_model(input_data_tensor, model_path)
    print("Predictions Shape:", predictions.shape)
    print("Predictions:", predictions)

    last_5_likes = list(map(int, user_input['last_5_likes'].split(',')))
    avg_last_5_likes = sum(last_5_likes) / len(last_5_likes)
    final_likes_prediction = avg_last_5_likes * predictions[0][0]

    prediction_result = {'likes': final_likes_prediction}

    return render_template('prediction_result.html', prediction=prediction_result)

def get_instagram_profiles(hashtag):
    conn = http.client.HTTPSConnection("instagram-scraper-api2.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "ccbb4992b3mshd0d7663e8db6e27p10302fjsnf8a0189bf83e",
        'x-rapidapi-host': "instagram-scraper-api2.p.rapidapi.com"
    }
    conn.request("GET", f"/v1/hashtag?hashtag={hashtag}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    profiles = []

    if res.status == 200:
        result = json.loads(data)
        items = result.get('data', {}).get('items', [])
        for item in items:
            user = item.get('user', {})
            username = user.get('username')
            if username:
                profile_url = f"https://www.instagram.com/{username}/"
                profiles.append(profile_url)
    else:
        print(f"Error: {res.status} - {res.reason}")
        print("Response data:", data.decode("utf-8"))

    return profiles

def generate_histogram(df, metrics):
    if not metrics:
        return None, [], []

    df['hour'] = pd.to_datetime(df['date']).dt.hour
    df['combined'] = df[metrics].prod(axis=1)
    df['relative'] = df['combined'] / df['combined'].max()
    grouped = df.groupby('hour')['relative'].mean()

    ax = grouped.plot(kind='bar', figsize=(10, 6))
    ax.set_xlabel('Година')
    ax.set_ylabel('Відносне значення')
    plt.title('Середнє відносне значення')
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'histogram.png')
    plt.savefig(plot_path)
    plt.close()

    top_hashtags = df['hashtags'].str.split(expand=True).stack().value_counts().head(10)
    top_words = df['caption'].str.split(expand=True).stack().value_counts().head(10)

    return plot_path, top_hashtags, top_words

@app.route('/plot_histogram.png')
def plot_histogram_png():
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'histogram.png')
    return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
