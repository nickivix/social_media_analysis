import instaloader
import pandas as pd
from datetime import datetime
import os
import yaml

def load_config(config_path='config.yaml'):
    base_path = os.path.abspath(os.path.dirname(__file__))
    full_path = os.path.join(base_path, '..', config_path)
    with open(full_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_instagram_posts(username, post_count=50):
    L = instaloader.Instaloader()
    posts_data = []

    try:
        profile = instaloader.Profile.from_username(L.context, username)
        posts = profile.get_posts()

        for i, post in enumerate(posts):
            if i >= post_count:
                break
            post_info = {
                'username': username,
                'date': post.date.strftime('%Y-%m-%d %H:%M:%S'),
                'likes': post.likes,
                'comments': post.comments,
                'views': post.video_view_count if post.is_video else 0,
                'hashtags': ' '.join(post.caption_hashtags),
                'caption': post.caption
            }
            posts_data.append(post_info)
    except Exception as e:
        print(f"Error fetching data for {username}: {e}")

    return posts_data

def collect_data_from_file(file_path):
    print(f"Collecting data: {file_path}")
    with open(file_path, 'r') as file:
        usernames = [line.strip().split('/')[-2] for line in file.readlines()]

    config = load_config()
    all_posts = []
    for username in usernames:
        print(f"Fetching posts for {username}")
        posts = get_instagram_posts(username, config['instagram']['post_count'])
        all_posts.extend(posts)

    if all_posts:
        df = pd.DataFrame(all_posts)
        base_path = os.path.abspath(os.path.dirname(__file__))
        data_path = os.path.join(base_path, '..', config['data']['raw_data_path'])
        df.to_csv(data_path, index=False)
        print(f"Data saved to {data_path}")
        return df
    else:
        print("No data")
        return None

if __name__ == "__main__":
    file_paths = [
        os.path.join('..', 'data', 'raw', 'popular_accounts.txt'),
        os.path.join('..', 'data', 'raw', 'searched_profiles.txt')
    ]

    for file_path in file_paths:
        if os.path.exists(file_path):
            collect_data_from_file(file_path)
        else:
            print(f"File {file_path} does not exist")
