#!/usr/bin/env bash
# One-shot setup for a fresh RunPod pod.
# Captures every workaround we hit during the first deployment:
#   - apt ffmpeg
#   - blinker uninstall conflict (Ubuntu apt-installed blinker)
#   - torch/torchvision version pin (xfuser may try to upgrade torch)
#   - diffusers 0.37 incompatible with torch 2.4 (FA3 op registration)
#   - peft must be >=0.17 for diffusers
#   - flash_attn is optional and often fails to build
#   - edge-tts 6.x is blocked by Microsoft, must be >=7.x
# Run once after cloning, before scripts/runpod_start.sh.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> 1/6 ffmpeg (system)"
apt-get update -qq
apt-get install -y -qq ffmpeg psmisc

echo "==> 2/6 work around apt-installed blinker that pip can't uninstall"
pip install -q --ignore-installed blinker

echo "==> 3/6 base requirements"
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r server/requirements-server.txt
pip install -q "huggingface_hub[cli]" hf_transfer

echo "==> 4/6 reinstall matched torch+torchvision (xfuser may have upgraded torch)"
pip install -q --force-reinstall --no-deps \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

echo "==> 5/6 fix dep mismatches we discovered the hard way"
# diffusers 0.37 registers FA3 ops that need torch 2.5+ -> downgrade
pip install -q "diffusers==0.31.0"
# peft must be >=0.17 for diffusers >=0.31
pip install -q -U "peft>=0.17.0"
# edge-tts 6.x is 403'd by Microsoft from data-center IPs
pip install -q -U edge-tts
# flash_attn is optional; ignore failures
pip install -q flash_attn --no-build-isolation 2>/dev/null || echo "  (flash_attn skipped, optional)"

echo "==> 6/6 sanity check"
python - <<'PY'
import torch, torchvision, fastapi, edge_tts, librosa, imageio, soundfile, yaml, peft, transformers, xfuser, diffusers
print(f"torch:        {torch.__version__}")
print(f"torchvision:  {torchvision.__version__}")
print(f"diffusers:    {diffusers.__version__}")
print(f"peft:         {peft.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"edge-tts:     {edge_tts.__version__ if hasattr(edge_tts, '__version__') else '7.x'}")
print(f"CUDA:         {torch.cuda.is_available()}  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
PY

echo ""
echo "Setup complete."
echo "Next:"
echo "  export HF_TOKEN=hf_xxx          # https://huggingface.co/settings/tokens"
echo "  bash scripts/download_weights.sh"
echo "  bash scripts/runpod_start.sh"
