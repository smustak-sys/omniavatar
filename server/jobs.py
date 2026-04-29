import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

STORAGE_ROOT = os.environ.get("OMNI_STORAGE", "/workspace/storage")
JOBS_DIR = os.path.join(STORAGE_ROOT, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

_lock = threading.Lock()


@dataclass
class Job:
    job_id: str
    status: str = "queued"  # queued | running | done | failed
    progress: float = 0.0
    message: str = ""
    video_path: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def path(self) -> str:
        return os.path.join(JOBS_DIR, f"{self.job_id}.json")

    def save(self):
        self.updated_at = datetime.utcnow().isoformat()
        with _lock:
            with open(self.path(), "w") as f:
                json.dump(asdict(self), f)

    @classmethod
    def load(cls, job_id: str) -> Optional["Job"]:
        p = os.path.join(JOBS_DIR, f"{job_id}.json")
        if not os.path.exists(p):
            return None
        with open(p, "r") as f:
            data = json.load(f)
        return cls(**data)


def new_job() -> Job:
    j = Job(job_id=uuid.uuid4().hex)
    j.save()
    return j
