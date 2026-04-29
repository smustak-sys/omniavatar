# OmniAvatar on RunPod — Deployment Runbook

This fork wraps [Omni-Avatar/OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar)
with a FastAPI service: upload an **image** + a **script** + a **gender**, the
server runs TTS (edge-tts) then OmniAvatar inference, and returns an mp4.

> **Note on lip sync:** OmniAvatar's strength is full-body / head motion. Lip
> sync is approximate, not frame-perfect. If frame-tight lip sync is critical,
> you want a video-input model like MuseTalk or LatentSync, not this.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/generate` | multipart: `image`, `script`, `gender` (male/female), `prompt` (optional) → `{job_id}` |
| `GET`  | `/status/{job_id}` | `{status, progress, message, error, video_url}` (progress updates live during inference) |
| `GET`  | `/video/{job_id}` | streams the mp4 |
| `GET`  | `/docs` | Swagger UI |

`status` ∈ `queued | running | done | failed`.

## RunPod web-terminal runbook

### 1. Create the pod

* **Template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or any
  similar PyTorch 2.4 + CUDA 12.4 + Ubuntu 22.04 image).
* **GPU:** **A40 48GB**, **A6000 48GB**, **L40S 48GB**, or **A100 80GB**.
  Anything <40GB needs `num_persistent_param_in_dit=7000000000` in the yaml.
* **Volume Disk:** ≥ **80 GB**. Container disk: 20 GB.
* **Expose HTTP Port:** **8000**.

### 2. Get a HuggingFace token (one minute)

Without a token, the model download stalls under unauthenticated rate limits.
Create a Read token at <https://huggingface.co/settings/tokens>.

### 3. In the web terminal — three commands

```bash
cd /workspace
git clone -b deploy https://github.com/smustak-sys/omniavatar.git OmniAvatar
cd OmniAvatar

# 1. Install all deps + workarounds in one shot (~5 min)
bash scripts/setup_pod.sh

# 2. Download model weights (~35 GB, ~5–15 min with token)
export HF_TOKEN=hf_PASTE_YOUR_TOKEN_HERE
bash scripts/download_weights.sh

# 3. Start the API
bash scripts/runpod_start.sh
```

When you see `INFO: Uvicorn running on http://0.0.0.0:8000`, click **Connect →
HTTP Service [Port 8000]** in the RunPod UI. Append `/docs` to the URL → Swagger.

> **Leave the server terminal alone.** The inference subprocess streams its
> tqdm progress bar back into this same terminal as it runs, so you can watch
> generation progress live without opening anything else.

### 4. Test from Swagger

1. **POST /generate** → Try it out:
   - **image:** upload a clear front-facing portrait (jpg/png).
   - **script:** the text the avatar should speak.
   - **gender:** `male` or `female`.
   - **prompt:** optional; defaults to a generic talking-head prompt.
   - Execute → copy the returned `job_id`.
2. **GET /status/{job_id}** → live progress (`0.25 → 0.95` during inference).
3. When status is `done` → **GET /video/{job_id}** → **Download file**.

> **Don't submit a second job while one is running.** They will fight for the
> GPU and most likely OOM the pod, leaving stuck jobs behind.

### 5. Test from curl

```bash
BASE="https://<your-pod>-8000.proxy.runpod.net"

JOB=$(curl -s -X POST "$BASE/generate" \
  -F "image=@portrait.jpg" \
  -F "script=Hello world." \
  -F "gender=female" | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")

while true; do
  S=$(curl -s "$BASE/status/$JOB")
  echo "$S"
  echo "$S" | grep -qE '"(done|failed)"' && break
  sleep 10
done

curl -L "$BASE/video/$JOB" -o output.mp4
```

## Tunables

| Var | Default | Notes |
|---|---|---|
| `PORT` | `8000` | server port |
| `OMNI_CONFIG` | `configs/inference.yaml` | swap to `configs/inference_1.3B.yaml` for the smaller, faster model (lower quality) |
| `OMNI_NPROC` | `1` | bump to N if you have N GPUs and update `sp_size` in yaml |

`configs/inference.yaml` (current defaults in this fork — already tuned):

| Field | Default here | Effect |
|---|---|---|
| `num_steps` | `20` | denoising steps; 20 is a good speed/quality balance with TeaCache |
| `tea_cache_l1_thresh` | `0.10` | enables TeaCache, ~1.5× speedup |
| `audio_scale` | `5.0` | audio guidance strength; 4–6 recommended for stronger lip sync |
| `guidance_scale` | `4.5` | prompt guidance |
| `max_hw` | `720` | 480p output (model is 480p-only) |

For a fresh 2-second clip on A100, expect **~3 minutes** end-to-end.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Download stalls at `4/27` | `HF_TOKEN` not set, or stale `.lock` files. The `download_weights.sh` script auto-cleans locks and retries. |
| `Cannot uninstall blinker 1.4` | Run `pip install --ignore-installed blinker` first, or just rerun `setup_pod.sh`. |
| `RuntimeError: operator torchvision::nms does not exist` | Torch/torchvision version mismatch. `setup_pod.sh` reinstalls a matched pair. |
| `infer_schema(func): Parameter q has unsupported type torch.Tensor` | Latest `diffusers` registers FA3 ops needing torch 2.5+. We pin `diffusers==0.31.0` in setup. |
| `peft>=0.17.0 is required` | `setup_pod.sh` upgrades peft. |
| TTS fails with `403 WSServer` | Old edge-tts blocked by Microsoft. Setup pins to >=7.x. |
| Status stuck at `running` after kill | The runner thread didn't update because the server itself died. Restart server; recovery script in setup. |
| Video has 29 chunks for a short script | Audio file is unexpectedly long — check `ffprobe -i storage/jobs/<id>/audio.mp3`. Usually means script was longer than expected. |

## Caveats

* **480p only.**
* **One job at a time.** Submitting a second `POST /generate` while a first is running will deadlock the GPU; both fail.
* **edge-tts goes over the public internet.** Microsoft has been known to throttle/block from cloud IPs. If you see TTS 403s and the upgrade doesn't fix it, swap `server/tts.py` for piper or Coqui XTTS-v2.
* **First request after pod start** triggers a one-time CUDA kernel compile and weight load to VRAM (~60–90 sec on A100). Subsequent requests are faster.
