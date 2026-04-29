#!/usr/bin/env bash
# Download OmniAvatar 14B weights into ./pretrained_models.
# Resumable. Cleans up stale .lock files from earlier interrupted runs.
# Recommended: export HF_TOKEN=hf_xxx before running, or you'll be rate-limited.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set. Unauthenticated downloads stall a lot."
  echo "         Get a free Read token at https://huggingface.co/settings/tokens"
fi
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p pretrained_models

# Prefer new `hf` CLI; fall back to deprecated `huggingface-cli`.
if command -v hf >/dev/null 2>&1; then
  HF=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli download)
else
  pip install -q "huggingface_hub[cli]"
  HF=(hf download)
fi

# Clear stale locks left by previous interrupted runs.
find pretrained_models -name "*.lock" -delete 2>/dev/null || true

download() {
  local repo="$1" dest="$2"
  echo ">> $repo -> $dest"
  # Up to 3 retries; hf download is resumable so completed files are skipped.
  for i in 1 2 3; do
    if "${HF[@]}" "$repo" --local-dir "$dest"; then
      return 0
    fi
    echo "   attempt $i failed, retrying after lock cleanup..."
    find "$dest/.cache/huggingface/download" -name "*.lock" -delete 2>/dev/null || true
    sleep 5
  done
  echo "ERROR: failed to download $repo after 3 attempts" >&2
  return 1
}

download "Wan-AI/Wan2.1-T2V-14B"           "./pretrained_models/Wan2.1-T2V-14B"
download "facebook/wav2vec2-base-960h"     "./pretrained_models/wav2vec2-base-960h"
download "OmniAvatar/OmniAvatar-14B"       "./pretrained_models/OmniAvatar-14B"

echo ">> Done. Disk usage:"
du -sh pretrained_models/*
