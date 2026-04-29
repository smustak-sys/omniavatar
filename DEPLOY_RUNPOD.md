# OmniAvatar on RunPod — Deployment Runbook

This fork wraps the official [Omni-Avatar/OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar)
inference pipeline with a FastAPI service and Swagger UI so you can drive it with
`image + script + gender` instead of `image + audio + prompt`. Speech is
synthesised on-the-fly with [edge-tts](https://github.com/rany2/edge-tts) (no API
key, no extra GPU memory).

```
client ──► POST /generate (image, script, gender)
              │
              ▼
        edge-tts ─► audio.mp3
              │
              ▼
   torchrun scripts/inference.py  (OmniAvatar 14B)
              │
              ▼
        result_000_*_wav.mp4
              │
              ▼
client ◄── GET /video/{job_id}
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/generate` | multipart: `image`, `script`, `gender` (male/female), `prompt` (optional) → `{job_id}` |
| `GET`  | `/status/{job_id}` | `{status, progress, message, error, video_url}` |
| `GET`  | `/video/{job_id}` | streams the mp4 |
| `GET`  | `/docs` | Swagger UI |
| `GET`  | `/health` | liveness |

`status` ∈ `queued | running | done | failed`.

## RunPod web-terminal runbook

### 1. Pick a pod

* **Template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
  (any image with PyTorch 2.4 + CUDA 12.4 + Ubuntu 22.04 works).
* **GPU:** **A40 48GB** or **A6000 48GB** (~$0.39–0.79/hr). Anything <40GB
  needs `num_persistent_param_in_dit=7000000000` to fit.
* **Volume:** at least **80 GB** disk — the 14B base weights alone are ~30 GB.
* **Expose HTTP port:** **8000**.

### 2. Open the web terminal and run

```bash
# 1. Clone this repo
cd /workspace
git clone https://github.com/smustak-sys/omniavatar.git OmniAvatar
cd OmniAvatar

# 2. Install dependencies (PyTorch is already in the RunPod image, but pin it)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -r server/requirements-server.txt
pip install "huggingface_hub[cli]" hf_transfer
# Optional, big perf win if it builds:
pip install flash_attn --no-build-isolation || echo "flash_attn skipped (optional)"

# 3. Install ffmpeg (needed for muxing audio into the output mp4)
apt-get update && apt-get install -y ffmpeg

# 4. Download model weights (~35 GB total, takes ~10–20 min)
export HF_HUB_ENABLE_HF_TRANSFER=1
bash scripts/download_weights.sh

# 5. Start the API
chmod +x scripts/runpod_start.sh
bash scripts/runpod_start.sh
```

When you see `Uvicorn running on http://0.0.0.0:8000`, click the **Connect →
HTTP Service [Port 8000]** button in the RunPod UI. The Swagger UI is at
`<that-url>/docs`.

### 3. Test from Swagger

1. Open `…/docs`.
2. Expand `POST /generate` → **Try it out**.
3. Upload an image (a clear front-facing portrait works best).
4. Type a script, e.g. *"Hello, this is a test of the OmniAvatar pipeline."*
5. Pick `male` or `female`. Optionally a prompt like
   *"A person speaking calmly to the camera, indoor lighting, plain background."*
6. Execute → copy the `job_id`.
7. Poll `GET /status/{job_id}` until `status: done` (1–3 min on A40 at 25 steps).
8. Open `GET /video/{job_id}` (Execute → **Download file**).

### 4. Test from curl

```bash
BASE="https://<your-pod>-8000.proxy.runpod.net"

JOB=$(curl -s -X POST "$BASE/generate" \
  -F "image=@portrait.jpg" \
  -F "script=Hello from OmniAvatar." \
  -F "gender=female" | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")

while :; do
  S=$(curl -s "$BASE/status/$JOB")
  echo "$S"
  echo "$S" | grep -q '"done"' && break
  echo "$S" | grep -q '"failed"' && break
  sleep 5
done

curl -L "$BASE/video/$JOB" -o output.mp4
```

## Tunables (env vars)

| Var | Default | Notes |
|---|---|---|
| `PORT` | `8000` | server port |
| `OMNI_CONFIG` | `configs/inference.yaml` | swap to `configs/inference_1.3B.yaml` for the smaller model |
| `OMNI_NPROC` | `1` | bump to N if you have N GPUs and update `sp_size` in yaml to match |
| `OMNI_STORAGE` | `<repo>/storage` | where job artefacts and the final mp4 live |

To trade quality for speed, edit `configs/inference.yaml`:

* `num_steps: 25` (default 50) — ~2× faster.
* `tea_cache_l1_thresh: 0.10` — extra ~1.5× speed-up, slight quality loss.
* `num_persistent_param_in_dit: 7000000000` — needed if VRAM < 40 GB.

## Known caveats

* The model is **480p only**.
* edge-tts goes over the public internet — for fully-offline TTS, swap
  `server/tts.py` for Coqui XTTS-v2.
* First request after pod start triggers a one-time CUDA kernel compile; subsequent
  requests are faster.
* `flash_attn` install often fails on fresh pods — it is optional and the
  pipeline runs without it.
