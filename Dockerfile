FROM python:3.11-slim

ARG USER=docker

WORKDIR /app

RUN useradd -u 1000 -ms /bin/bash ${USER}
USER ${USER}

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    libsox-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


