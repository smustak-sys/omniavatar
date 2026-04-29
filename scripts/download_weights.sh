#!/usr/bin/env bash
# Download OmniAvatar 14B weights into ./pretrained_models
# Run from repo root.
set -euo pipefail

mkdir -p pretrained_models

# Newer huggingface_hub ships the `hf` command; older versions ship
# `huggingface-cli`. Prefer `hf` and fall back.
if command -v hf >/dev/null 2>&1; then
  HF="hf download"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF="huggingface-cli download"
else
  pip install -q "huggingface_hub[cli]"
  HF="hf download"
fi

echo ">> Using: $HF"

echo ">> Downloading Wan2.1-T2V-14B (base model, ~30GB)..."
$HF Wan-AI/Wan2.1-T2V-14B --local-dir ./pretrained_models/Wan2.1-T2V-14B

echo ">> Downloading wav2vec2-base-960h (audio encoder)..."
$HF facebook/wav2vec2-base-960h --local-dir ./pretrained_models/wav2vec2-base-960h

echo ">> Downloading OmniAvatar-14B (LoRA + audio cond)..."
$HF OmniAvatar/OmniAvatar-14B --local-dir ./pretrained_models/OmniAvatar-14B

echo ">> Done. Disk usage:"
du -sh pretrained_models/*
