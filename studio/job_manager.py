# studio/job_manager.py
"""In-memory job registry with per-UC locking for pipeline steps."""

import uuid
import time
from enum import Enum
from threading import Lock
from typing import Optional


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"
    STOPPED = "stopped"


class DuplicateJobError(Exception):
    """Raised when a UC already has a running job."""


_MAX_LOG_LINES = 100


class JobManager:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._uc_locks: dict[str, str] = {}  # uc_id -> job_id
        self._lock = Lock()

    def create(self, uc_id: str, step: str) -> str:
        with self._lock:
            if uc_id in self._uc_locks:
                raise DuplicateJobError(
                    f"{uc_id} already has a running job: {self._uc_locks[uc_id]}"
                )
            job_id = str(uuid.uuid4())[:8]
            self._jobs[job_id] = {
                "job_id":     job_id,
                "uc_id":      uc_id,
                "step":       step,
                "status":     JobStatus.RUNNING,
                "pid":        None,
                "start_time": time.time(),
                "last_lines": [],
                "error":      None,
            }
            self._uc_locks[uc_id] = job_id
            return job_id

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def set_pid(self, job_id: str, pid: int):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["pid"] = pid

    def append_log(self, job_id: str, line: str):
        with self._lock:
            if job_id in self._jobs:
                lines = self._jobs[job_id]["last_lines"]
                lines.append(line)
                if len(lines) > _MAX_LOG_LINES:
                    self._jobs[job_id]["last_lines"] = lines[-_MAX_LOG_LINES:]

    def complete(self, job_id: str, exit_code: int):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["status"] = JobStatus.DONE if exit_code == 0 else JobStatus.FAILED
                self._uc_locks.pop(job["uc_id"], None)

    def fail(self, job_id: str, error: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["status"] = JobStatus.FAILED
                job["error"] = error
                self._uc_locks.pop(job["uc_id"], None)

    def stop(self, job_id: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["status"] = JobStatus.STOPPED
                self._uc_locks.pop(job["uc_id"], None)

    def list_active(self) -> list[dict]:
        with self._lock:
            return [dict(j) for j in self._jobs.values() if j["status"] == JobStatus.RUNNING]

    def last_for_uc(self, uc_id: str) -> Optional[dict]:
        with self._lock:
            matches = [j for j in self._jobs.values() if j["uc_id"] == uc_id]
            if not matches:
                return None
            return dict(max(matches, key=lambda j: j["start_time"]))


# Module-level singleton used by routes and pipeline_runner
_manager = JobManager()

def get_manager() -> JobManager:
    return _manager
