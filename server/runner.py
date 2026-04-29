import glob
import os
import subprocess
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

        job.message = "Running OmniAvatar inference"
        job.progress = 0.25
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
        log_path = os.path.join(wd, "run.log")
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                cmd, cwd=REPO_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT
            )
        if proc.returncode != 0:
            with open(log_path, "r", errors="ignore") as logf:
                tail = logf.read()[-2000:]
            raise RuntimeError(f"inference failed (exit {proc.returncode}). Tail:\n{tail}")

        # find output mp4: demo_out/<exp_name>/res_input_*/result_000.mp4
        candidates = sorted(
            glob.glob(os.path.join(REPO_ROOT, "demo_out", "*", "res_input_*", "result_000*.mp4")),
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
        # copy to job dir so cleanup is local
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
