"""SyncScheduler Protocol and APScheduler-based backend."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol, runtime_checkable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from apscheduler.jobstores.base import JobLookupError as APSJobLookupError


@dataclass
class SyncJob:
    job_id: str
    uc_name: str
    interval_minutes: int
    last_run: datetime | None = None
    next_run: datetime | None = None
    last_status: str = "pending"


@runtime_checkable
class SyncScheduler(Protocol):
    def schedule(self, uc_name: str, interval_minutes: int, fn: Callable) -> str: ...
    def cancel(self, job_id: str) -> None: ...
    def list_jobs(self) -> list[SyncJob]: ...
    def trigger_now(self, job_id: str) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class APSchedulerBackend:
    """In-process sync scheduler backed by APScheduler."""

    def __init__(self):
        self._scheduler = BackgroundScheduler()
        self._jobs: dict[str, SyncJob] = {}
        self._scheduler.add_listener(self._on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    def _on_job_event(self, event) -> None:
        tracked = self._jobs.get(event.job_id)
        if tracked:
            tracked.last_run = datetime.now(timezone.utc)
            tracked.last_status = "error" if event.exception else "ok"

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    def schedule(self, uc_name: str, interval_minutes: int, fn: Callable) -> str:
        job_id = f"sync_{uc_name}"
        self._scheduler.add_job(
            fn,
            "interval",
            minutes=interval_minutes,
            id=job_id,
            replace_existing=True,
        )
        apjob = self._scheduler.get_job(job_id)
        self._jobs[job_id] = SyncJob(
            job_id=job_id,
            uc_name=uc_name,
            interval_minutes=interval_minutes,
            next_run=apjob.next_run_time if apjob else None,
        )
        return job_id

    def cancel(self, job_id: str) -> None:
        try:
            self._scheduler.remove_job(job_id)
        except APSJobLookupError:
            pass
        self._jobs.pop(job_id, None)

    def list_jobs(self) -> list[SyncJob]:
        for job_id, tracked in self._jobs.items():
            apjob = self._scheduler.get_job(job_id)
            if apjob:
                tracked.next_run = apjob.next_run_time
        return list(self._jobs.values())

    def trigger_now(self, job_id: str) -> None:
        try:
            self._scheduler.modify_job(job_id, next_run_time=datetime.now(timezone.utc))
            tracked = self._jobs.get(job_id)
            if tracked:
                tracked.last_run = datetime.now(timezone.utc)
                tracked.last_status = "ok"
        except Exception:
            pass
