FROM python:3.11-slim

WORKDIR /app

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    libsox-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 依存関係をコピー・インストール
COPY requirements.txt ./
COPY gui/requirements_gui.txt ./gui/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r gui/requirements_gui.txt

# プロジェクトファイルをコピー
COPY . .

# ポート公開
EXPOSE 7860

# GUIアプリケーション起動
WORKDIR /app/gui
CMD ["python", "launch.py"]
