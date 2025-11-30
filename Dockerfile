FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    ffmpeg \
    git \
    build-essential \
    libsox-dev \
    libsndfile1 \
    curl \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set python as alias for python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install PyTorch 2.5.1 with CUDA 12.4 support
# mamba-ssm has prebuilt wheels for this version
RUN pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Override laion-clap version (stable-audio-tools requires 1.1.4, but we want 1.1.7)
RUN pip install --no-cache-dir --no-deps --force-reinstall laion-clap==1.1.7
