# sync/scheduler.py
"""APScheduler-based cron scheduler for connector syncs.

Started in FastAPI lifespan; reads connector_configs.schedule_cron on startup.
"""
from __future__ import annotations
from typing import Callable
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import db.store as store


class SyncScheduler:
    def __init__(self, run_fn: Callable):
        """run_fn(connector_id, uc_id, trigger) — coroutine or plain callable."""
        self._run_fn = run_fn
        self._scheduler = AsyncIOScheduler()
        self._job_ids: dict[str, list[str]] = {}  # connector_id → [job_id, ...]

    def load_from_db(self) -> None:
        rows = store.fetchall(
            "SELECT cc.id as cid, cc.schedule_cron, cus.uc_id "
            "FROM connector_configs cc "
            "JOIN connector_uc_scopes cus ON cus.connector_id=cc.id "
            "WHERE cc.schedule_cron IS NOT NULL"
        )
        for row in rows:
            self._add_job(row["cid"], row["uc_id"], row["schedule_cron"])

    def _add_job(self, connector_id: str, uc_id: str, cron_expr: str) -> str:
        parts = cron_expr.split()
        if len(parts) == 5:
            minute, hour, day, month, dow = parts
        else:
            minute, hour, day, month, dow = "*", "*", "*", "*", "*"
        trigger = CronTrigger(
            minute=minute, hour=hour, day=day, month=month, day_of_week=dow
        )
        job_id = f"sync_{connector_id}_{uc_id}"
        self._scheduler.add_job(
            self._run_fn, trigger,
            args=[connector_id, uc_id, "scheduled"],
            id=job_id, replace_existing=True,
        )
        self._job_ids.setdefault(connector_id, [])
        if job_id not in self._job_ids[connector_id]:
            self._job_ids[connector_id].append(job_id)
        return job_id

    def reschedule(self, connector_id: str, cron_expr: str) -> None:
        self.remove(connector_id)
        scopes = store.fetchall(
            "SELECT uc_id FROM connector_uc_scopes WHERE connector_id=?",
            (connector_id,)
        )
        for row in scopes:
            self._add_job(connector_id, row["uc_id"], cron_expr)

    def remove(self, connector_id: str) -> None:
        for jid in self._job_ids.pop(connector_id, []):
            try:
                self._scheduler.remove_job(jid)
            except Exception:
                pass

    def list_jobs(self) -> list[dict]:
        result = []
        for cid, jids in self._job_ids.items():
            for jid in jids:
                job = self._scheduler.get_job(jid)
                next_run = getattr(job, "next_run_time", None) if job else None
                result.append({
                    "connector_id": cid,
                    "job_id": jid,
                    "next_run": str(next_run) if next_run else None,
                })
        return result

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
