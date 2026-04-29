FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git ffmpeg curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /workspace/OmniAvatar
COPY requirements.txt server/requirements-server.txt ./

RUN pip install --upgrade pip && \
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements.txt && \
    pip install -r requirements-server.txt && \
    pip install "huggingface_hub[cli]" hf_transfer

COPY . .

ENV OMNI_REPO_ROOT=/workspace/OmniAvatar \
    OMNI_STORAGE=/workspace/storage \
    OMNI_CONFIG=/workspace/OmniAvatar/configs/inference.yaml \
    OMNI_NPROC=1 \
    PYTHONPATH=/workspace/OmniAvatar

RUN chmod +x scripts/runpod_start.sh scripts/download_weights.sh

EXPOSE 8000
CMD ["bash", "scripts/runpod_start.sh"]
