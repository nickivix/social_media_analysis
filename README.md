# Шуліка Назар ПЗ-20-1

## Опис

Проект спрямований на прогнозування поведінки користувачів у соціальних мережах за допомогою нейронної мережі типу GRU.

## Структура проекту

- `config.yaml`: конфігураційний файл.
- `README.md`: опис проекту.
- `requirements.txt`: список залежностей.
- `.idea/`: налаштування IDE.
- `data/`: директорія для даних.
  - `external/`: зовнішні джерела даних.
  - `processed/`: оброблені дані.
    - `X_test.csv`
    - `X_train.csv`
    - `X_val.csv`
    - `y_test.csv`
    - `y_train.csv`
    - `y_val.csv`
  - `raw/`: сирі дані.
    - `instagram_global_top_1000.csv`
    - `instagram_posts.csv`
    - `popular_accounts.csv`
    - `popular_accounts.txt`
    - `searched_accounts.txt`
- `notebooks/`: Jupyter ноутбуки для дослідження даних.
  - `exploratory_data_analysis.ipynb`
- `results/`: результати роботи моделі.
  - `logs/`: логи навчання.
  - `models/`: збережені моделі.
    - `model.h5`
    - `caption_vectorizer.pkl`
    - `hashtag_vectorizer.pkl`
  - `plots/`: графіки результатів.
- `src/`: вихідний код.
  - `data_preprocessing.py`: модуль для підготовки даних.
  - `evaluate.py`: модуль для оцінки моделі.
  - `instagram_data_collector.py`: модуль для збору даних з Instagram.
  - `model.py`: модуль для навчання, визначення та використання моделі.
  - `train.py`: модуль для навчання моделі.
  - `utils.py`: додаткові утиліти.
  - `web_app/`: директорія веб-додатку.
    - `app.py`: основний файл веб-додатку.
    - `static/`: статичні файли для веб-додатку.
      - `css/`
        - `styles.css`
      - `js/`
        - `scripts.js`
    - `templates/`: шаблони HTML для веб-додатку.
      - `existing_data.html`
      - `index.html`
      - `prediction_result.html`
  - `__pycache__/`: кешовані файли Python.
- `venv/`: віртуальне середовище.

## Інсталяція

1. Клонувати репозиторій:
    ```bash
    git clone <URL>
    cd project
    ```

2. Встановити залежності:
    ```bash
    pip install -r requirements.txt
    ```

## Використання

1. Підготовка даних:
    ```bash
    python src/data_preprocessing.py
    ```

2. Навчання моделі:
    ```bash
    python src/model.py
    ```

3. Запуск Веб-Додатка:
    ```bash
    python src/web_app/app.py
    ```

## Конфігурація

Вся конфігурація зберігається у файлі `config.yaml`. Ви можете змінити налаштування моделі, шляхи до даних та параметри навчання, редагуючи цей файл.

## Передбачення

1. Веб-інтерфейс дозволяє користувачам вводити наступну інформацію для передбачення популярності посту:
    - Час дня (0-23)
    - Хештеги (розділені пробілами)
    - Опис
    - Лайки за останні 5 постів (розділені комами)

2. Введені дані обробляються, передбачення виконуються нейронною мережею, і користувач бачить передбачувані лайки.
