
#run command - 

# docker run -it --rm --gpus all --ipc=host [build docker image ]
#docker run -it --gpus all --ipc=host -v $HOME/vidur-simulator:$HOME/vidur-simulator/ -v ~/.cache/huggingface:/root/.cache/huggingface -e RAY_CGRAPH_get_timeout=300 -e RAY_CGRAPH_submit_timeout=300 -e HF_TOKEN nba556677/vllm:12.8 bash
# Start with NVIDIA CUDA 12.8 base image
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install basic Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel pandas

# Install PyTorch with CUDA support
# Using PyTorch 2.2.x which is compatible with CUDA 12.x
RUN python3 -m pip install --no-cache-dir torch torchvision torchaudio

# Install vLLM
RUN python3 -m pip install --no-cache-dir vllm

#install ray
RUN python3 -m pip install --no-cache-dir ray

# Create a working directory
WORKDIR /app

# Set the default command
CMD ["/bin/bash"]



