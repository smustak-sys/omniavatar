import glob
import os
import re
import subprocess
import sys
import threading
import traceback

from .jobs import Job, STORAGE_ROOT
from . import tts

REPO_ROOT = os.environ.get("OMNI_REPO_ROOT", "/workspace/OmniAvatar")
CONFIG_PATH = os.environ.get(
    "OMNI_CONFIG", os.path.join(REPO_ROOT, "configs/inference.yaml")
)
NPROC = int(os.environ.get("OMNI_NPROC", "1"))


def _job_dir(job_id: str) -> str:
    d = os.path.join(STORAGE_ROOT, "jobs", job_id)
    os.makedirs(d, exist_ok=True)
    return d


# Matches the tqdm bar lines we want to use for progress. Examples:
#   "  0%|          | 0/25 [00:00<?, ?it/s]"
#   " 40%|████      | 10/25 [02:48<04:11, 16.78s/it]"
_TQDM_RE = re.compile(r"\b(\d+)/(\d+)\s+\[")
# Matches "[t/T]" chunk markers from inference.py.
_CHUNK_RE = re.compile(r"^\[(\d+)/(\d+)\]\s*$")


def _stream_subprocess(cmd, cwd, env, log_path, job: Job):
    """Run cmd, stream output line-by-line to (a) the log file, (b) stdout
    (so the uvicorn terminal shows progress), and (c) parse tqdm bars to
    update job.progress in near-real-time."""
    chunk_idx, chunk_total = 1, 1
    with open(log_path, "w", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            errors="replace",
        )
        last_progress = -1.0
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                # mirror to log file and uvicorn stdout
                logf.write(line)
                sys.stdout.write(line)
                sys.stdout.flush()

                # update progress
                m = _CHUNK_RE.match(line.strip())
                if m:
                    chunk_idx = int(m.group(1))
                    chunk_total = int(m.group(2))
                    continue

                m = _TQDM_RE.search(line)
                if m:
                    step, total_steps = int(m.group(1)), int(m.group(2))
                    if total_steps == 0:
                        continue
                    # 0.25 .. 0.95 reserved for inference; split across chunks
                    chunk_frac = (chunk_idx - 1 + step / total_steps) / max(chunk_total, 1)
                    p = 0.25 + 0.70 * chunk_frac
                    p = round(min(max(p, 0.25), 0.95), 3)
                    if p > last_progress + 0.01:
                        job.progress = p
                        job.message = f"Inference chunk {chunk_idx}/{chunk_total}, step {step}/{total_steps}"
                        job.save()
                        last_progress = p
        finally:
            proc.wait()
        return proc.returncode


def run_job(job_id: str, image_path: str, script: str, gender: str, prompt: str):
    job = Job.load(job_id)
    if job is None:
        return
    try:
        job.status = "running"
        job.message = "Synthesizing speech"
        job.progress = 0.05
        job.save()

        wd = _job_dir(job_id)
        audio_path = os.path.join(wd, "audio.mp3")
        tts.synthesize(script, gender, audio_path)

        job.message = "Preparing input"
        job.progress = 0.15
        job.save()

        input_file = os.path.join(wd, "input.txt")
        prompt = (prompt or "A person speaking to the camera, natural lighting, clear background").strip()
        with open(input_file, "w") as f:
            f.write(f"{prompt}@@{image_path}@@{audio_path}\n")

        job.message = "Loading model"
        job.progress = 0.20
        job.save()

        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={NPROC}",
            "scripts/inference.py",
            "--config",
            CONFIG_PATH,
            "--input_file",
            input_file,
        ]
        env = os.environ.copy()
        # ensure tqdm prints unbuffered so we can read line-by-line
        env.setdefault("PYTHONUNBUFFERED", "1")
        log_path = os.path.join(wd, "run.log")

        rc = _stream_subprocess(cmd, REPO_ROOT, env, log_path, job)
        if rc != 0:
            with open(log_path, "r", errors="ignore") as logf:
                tail = logf.read()[-2000:]
            raise RuntimeError(f"inference failed (exit {rc}). Tail:\n{tail}")

        job.message = "Finalising video"
        job.progress = 0.97
        job.save()

        # Find output mp4: prefer the *_wav.mp4 (audio muxed)
        candidates = sorted(
            glob.glob(os.path.join(REPO_ROOT, "demo_out", "*", "res_input_*", "result_*_wav.mp4")),
            key=os.path.getmtime,
            reverse=True,
        )
        if not candidates:
            candidates = sorted(
                glob.glob(os.path.join(REPO_ROOT, "demo_out", "**", "*.mp4"), recursive=True),
                key=os.path.getmtime,
                reverse=True,
            )
        if not candidates:
            raise RuntimeError("No mp4 produced in demo_out/")

        final_path = os.path.join(wd, "output.mp4")
        import shutil

        shutil.copy2(candidates[0], final_path)

        job.video_path = final_path
        job.status = "done"
        job.progress = 1.0
        job.message = "Completed"
        job.save()
    except Exception as e:
        job.status = "failed"
        job.error = f"{e}\n{traceback.format_exc()}"
        job.message = "Failed"
        job.save()


def run_job_async(job_id: str, image_path: str, script: str, gender: str, prompt: str):
    t = threading.Thread(
        target=run_job,
        args=(job_id, image_path, script, gender, prompt),
        daemon=True,
    )
    t.start()
