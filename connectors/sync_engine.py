# connectors/sync_engine.py
"""Orchestrates list→diff→download→ingest→upsert per connector+UC scope."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Callable

import db.store as store
from connectors.base import SourceFile
from connectors.registry import ConnectorRegistry
from sync.progress import publish

_registry = ConnectorRegistry()


class SyncEngine:
    def __init__(
        self,
        connector_factory: Callable,   # (type: str, creds: dict) -> SourceConnector
        embed_fn: Callable,             # (chunks: list[str]) -> list[list[float]]
        upsert_fn: Callable,            # (uc_id, chunks, embeddings, kv_tensors) -> None
        kv_fn: Callable,                # (chunks: list[str]) -> list[tensor|None]
    ):
        self._factory = connector_factory
        self._embed = embed_fn
        self._upsert = upsert_fn
        self._kv = kv_fn

    async def run(self, connector_id: str, uc_id: str, trigger: str) -> str:
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        store.execute(
            "INSERT INTO sync_runs(id,connector_id,uc_id,trigger,status,started_at) VALUES(?,?,?,?,?,?)",
            (run_id, connector_id, uc_id, trigger, "running", now)
        )
        store.commit()

        try:
            await self._run_inner(run_id, connector_id, uc_id, trigger)
            store.execute(
                "UPDATE sync_runs SET status='ok',finished_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), run_id)
            )
            store.commit()
            await publish(connector_id, {"event": "complete", "connector_id": connector_id,
                                         "uc_id": uc_id, "run_id": run_id})
        except Exception as exc:
            store.execute(
                "UPDATE sync_runs SET status='error',error=?,finished_at=? WHERE id=?",
                (str(exc), datetime.now(timezone.utc).isoformat(), run_id)
            )
            store.commit()
            await publish(connector_id, {"event": "error", "connector_id": connector_id,
                                         "uc_id": uc_id, "error": str(exc)})
        return run_id

    async def _run_inner(self, run_id: str, connector_id: str, uc_id: str, trigger: str) -> None:
        creds = _registry.get_credentials(connector_id)
        cfg = _registry.get(connector_id)
        scopes = _registry.list_scopes(connector_id)
        scope = next((s for s in scopes if s["uc_id"] == uc_id), None)
        if scope is None:
            raise ValueError(f"no scope configured for connector {connector_id} / UC {uc_id}")

        connector = self._factory(cfg["type"], creds)

        await publish(connector_id, {"event": "progress", "stage": "discover",
                                     "connector_id": connector_id, "uc_id": uc_id,
                                     "files_total": 0, "files_done": 0,
                                     "message": "discovering files…"})

        loop = asyncio.get_running_loop()
        if connector.supports_delta():
            last_token = scope.get("last_delta_token")
            files, new_token = await loop.run_in_executor(
                None, connector.get_delta, last_token)
        else:
            all_files: list[SourceFile] = await loop.run_in_executor(None, connector.list_files)
            scope_row = store.fetchone(
                "SELECT last_sync_at FROM connector_uc_scopes WHERE connector_id=? AND uc_id=?",
                (connector_id, uc_id)
            )
            last_sync = scope_row["last_sync_at"] if scope_row and scope_row["last_sync_at"] else None
            if last_sync:
                last_dt = datetime.fromisoformat(last_sync.replace("Z", "")).replace(tzinfo=None)
                files = [f for f in all_files if f.modified_at.replace(tzinfo=None) > last_dt]
            else:
                files = all_files
            new_token = None

        store.execute("UPDATE sync_runs SET files_total=? WHERE id=?", (len(files), run_id))
        store.commit()
        await publish(connector_id, {"event": "progress", "stage": "discover",
                                     "connector_id": connector_id, "uc_id": uc_id,
                                     "files_total": len(files), "files_done": 0,
                                     "message": f"{len(files)} files to process"})

        for i, sf in enumerate(files):
            raw: bytes = await loop.run_in_executor(None, connector.download, sf)
            chunks = _chunk_bytes(sf, raw)

            await publish(connector_id, {"event": "progress", "stage": "index",
                                         "connector_id": connector_id, "uc_id": uc_id,
                                         "files_total": len(files), "files_done": i,
                                         "message": f"embedding {sf.name}"})

            if chunks:
                embeddings = self._embed(chunks)
                kv_tensors = self._kv(chunks)
                self._upsert(uc_id, chunks, embeddings, kv_tensors)

            store.execute("UPDATE sync_runs SET files_done=? WHERE id=?", (i + 1, run_id))
            store.commit()

        now = datetime.now(timezone.utc).isoformat()
        store.execute(
            "UPDATE connector_uc_scopes SET last_sync_at=?,last_delta_token=? "
            "WHERE connector_id=? AND uc_id=?",
            (now, new_token, connector_id, uc_id)
        )
        store.commit()


def _chunk_bytes(sf: SourceFile, raw: bytes) -> list[str]:
    """Simple line-based chunking; replace with loader dispatch if needed."""
    text = raw.decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks, buf = [], ""
    for line in lines:
        if len(buf) + len(line) > 512:
            if buf:
                chunks.append(buf)
            buf = line
        else:
            buf = buf + " " + line if buf else line
    if buf:
        chunks.append(buf)
    return chunks


def _default_connector_factory(connector_type: str, creds: dict):
    """Build a real SourceConnector from type + decrypted credentials."""
    if connector_type == "gdrive":
        from connectors.gdrive_connector import GDriveConnector
        return GDriveConnector(creds)
    if connector_type == "s3":
        from connectors.s3_connector import S3Connector
        return S3Connector(creds)
    if connector_type == "sharepoint":
        from connectors.sharepoint_connector import SharePointConnector
        return SharePointConnector(creds)
    raise ValueError(f"unknown connector type: {connector_type}")


def make_default_engine() -> SyncEngine:
    """Create a SyncEngine wired to real connectors (no-op embed/upsert/kv for now)."""
    return SyncEngine(
        connector_factory=_default_connector_factory,
        embed_fn=lambda chunks: [[0.0] * 384 for _ in chunks],
        upsert_fn=lambda uc_id, chunks, embs, kvs: None,
        kv_fn=lambda chunks: [None] * len(chunks),
    )
