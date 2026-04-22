FROM python:3.11-slim

WORKDIR /app

# 系統依賴（spacy 編譯需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 升級 pip
RUN pip install --upgrade pip

# 第一層：先裝最耗時的大套件（Docker 會快取這層）
RUN pip install spacy==3.7.4 lemminflect==0.2.3

# 第二層：下載 spacy 語言模型
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# 第三層：裝其他輕量套件
RUN pip install \
    django==4.2.16 \
    pandas==2.2.2 \
    whitenoise==6.7.0 \
    gunicorn==22.0.0

# 複製專案程式碼（放最後，避免程式碼變動讓前面快取失效）
COPY . .

# 收集靜態檔案（Django）
RUN python manage.py collectstatic --noinput || true

EXPOSE 8000

CMD gunicorn toeic.wsgi --bind 0.0.0.0:$PORT
