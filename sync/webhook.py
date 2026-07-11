# sync/webhook.py
"""HMAC-SHA256 validated webhook receiver for push change notifications.

Accepts POST /webhooks/{connector_id} — no auth cookie required.
Validates X-Hub-Signature-256 header; enqueues sync as background task.
"""
import hashlib
import hmac
from typing import Callable

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse
import db.store as store

# Module-level router (unused in tests; use make_webhook_router for testability)
webhook_router = APIRouter(tags=["webhooks"])


def make_webhook_router(run_fn: Callable) -> APIRouter:
    """Factory so tests can inject a fake run_fn."""
    router = APIRouter(tags=["webhooks"])

    @router.post("/webhooks/{connector_id}")
    async def receive_webhook(connector_id: str, request: Request,
                              background_tasks: BackgroundTasks):
        row = store.fetchone(
            "SELECT webhook_secret FROM connector_configs WHERE id=?", (connector_id,))
        if not row:
            return JSONResponse({"detail": "connector not found"}, status_code=404)

        secret = row["webhook_secret"] or ""
        body = await request.body()
        sig_header = request.headers.get("X-Hub-Signature-256", "")
        expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig_header, expected):
            return JSONResponse({"detail": "invalid signature"}, status_code=401)

        scopes = store.fetchall(
            "SELECT uc_id FROM connector_uc_scopes WHERE connector_id=?", (connector_id,))
        for scope in scopes:
            background_tasks.add_task(run_fn, connector_id, scope["uc_id"], "webhook")

        return {"ok": True, "queued_scopes": len(scopes)}

    return router
