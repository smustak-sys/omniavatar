import os
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .jobs import Job, STORAGE_ROOT, new_job
from .runner import run_job_async

app = FastAPI(
    title="OmniAvatar API",
    description=(
        "Audio-driven avatar video generation.\n\n"
        "Upload a single image of a person, provide a script (the words they should "
        "say) and a gender (used to pick the TTS voice). The server synthesises "
        "speech, runs OmniAvatar inference, and returns an mp4."
    ),
    version="1.0.0",
)


@app.get("/", tags=["meta"])
def root():
    return {"name": "OmniAvatar API", "docs": "/docs"}


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.post("/generate", tags=["video"])
async def generate(
    image: UploadFile = File(..., description="Reference image (jpg/png) of the person"),
    script: str = Form(..., description="Text the avatar should speak"),
    gender: Literal["male", "female"] = Form("female", description="TTS voice gender"),
    prompt: Optional[str] = Form(
        None,
        description="Optional scene/style prompt. Defaults to a generic talking-head prompt.",
    ),
):
    """Submit a generation job. Returns a job_id you can poll on /status/{job_id}."""
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "image must be jpeg or png")

    job = new_job()
    job_dir = os.path.join(STORAGE_ROOT, "jobs", job.job_id)
    os.makedirs(job_dir, exist_ok=True)

    ext = ".png" if image.content_type == "image/png" else ".jpg"
    image_path = os.path.join(job_dir, f"input{ext}")
    with open(image_path, "wb") as f:
        f.write(await image.read())

    run_job_async(job.job_id, image_path, script, gender, prompt or "")

    return {"job_id": job.job_id, "status": job.status}


@app.get("/status/{job_id}", tags=["video"])
def status(job_id: str):
    """Get current status of a job."""
    job = Job.load(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
        "video_url": f"/video/{job.job_id}" if job.status == "done" else None,
    }


@app.get("/video/{job_id}", tags=["video"])
def video(job_id: str):
    """Download the generated mp4."""
    job = Job.load(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != "done" or not job.video_path or not os.path.exists(job.video_path):
        raise HTTPException(409, f"video not ready (status={job.status})")
    return FileResponse(
        job.video_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
    )
