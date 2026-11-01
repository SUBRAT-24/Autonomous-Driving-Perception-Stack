# ==============================================================================
# Autonomous Driving Perception Stack — Production & Demo Dockerfile
# Supports NVIDIA GPU acceleration (CUDA 12.1) and CPU fallback
# ==============================================================================

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive timezone and keyboard prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install base system utilities and graphics libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Upgrade pip and wheel
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN python -m pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copy repository dependency files
COPY pyproject.toml requirements.txt /app/

# Install repository dependencies including Gradio
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir gradio

# Copy application source code
COPY . /app

# Expose default Gradio web port
EXPOSE 7860

# Default entrypoint: launch interactive Gradio web application
CMD ["python", "app.py"]

