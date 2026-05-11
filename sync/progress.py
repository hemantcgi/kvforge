# sync/progress.py
"""Per-connector asyncio.Queue SSE bus.

SyncEngine publishes dicts; GET /sync/stream/{connector_id} drains them.
"""
import asyncio
import json
from collections import defaultdict
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

_queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

progress_router = APIRouter(tags=["sync-progress"])


class ProgressBus:
    """Thin wrapper so tests can instantiate an isolated bus."""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

    async def publish(self, connector_id: str, event: dict) -> None:
        await self._queues[connector_id].put(event)

    async def get(self, connector_id: str) -> dict:
        return await self._queues[connector_id].get()


# Module-level singleton used by SyncEngine and the SSE route
_bus = ProgressBus()


async def publish(connector_id: str, event: dict) -> None:
    await _bus.publish(connector_id, event)


@progress_router.get("/sync/stream/{connector_id}")
async def sync_stream(connector_id: str, request: Request):
    u = getattr(request.state, "user", None)
    if not u or u.role not in ("admin", "editor"):
        return JSONResponse({"detail": "forbidden"}, status_code=403)

    async def event_gen():
        q = _bus._queues[connector_id]
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("event") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})
