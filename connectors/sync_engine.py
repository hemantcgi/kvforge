# connectors/sync_engine.py
"""Orchestrates list→diff→download→ingest→upsert per connector+UC scope.

After each successful sync that produces new chunks the engine fires the
post-sync pipeline automatically:

  1. FAQ generation  (sleep_faq_generator — incremental, skips already-done chunks)
  2. Recompute KV    (kv_indexer compute-kv --filter kv_version=null)

LoRA training and PRS evaluation are intentionally NOT triggered automatically
because they are expensive GPU operations.  They remain user-initiated from
the Studio Pipeline panel.  The sync pipeline focuses on keeping the vector
store and KV cache fresh so queries stay accurate while training is scheduled
separately.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import db.store as store
from connectors.base import SourceFile
from connectors.registry import ConnectorRegistry
from sync.progress import publish

_registry = ConnectorRegistry()
ROOT = Path(__file__).resolve().parent.parent


class SyncEngine:
    def __init__(
        self,
        connector_factory: Callable,   # (type: str, creds: dict) -> SourceConnector
        embed_fn: Callable,             # (uc_id: str, chunks: list[str]) -> list[list[float]]
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
            new_chunk_count = await self._run_inner(run_id, connector_id, uc_id, trigger)
            store.execute(
                "UPDATE sync_runs SET status='ok',finished_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), run_id)
            )
            store.commit()
            await publish(connector_id, {"event": "complete", "connector_id": connector_id,
                                         "uc_id": uc_id, "run_id": run_id,
                                         "new_chunks": new_chunk_count})
            if new_chunk_count > 0:
                asyncio.create_task(
                    self._post_sync_pipeline(connector_id, uc_id, new_chunk_count)
                )
        except Exception as exc:
            store.execute(
                "UPDATE sync_runs SET status='error',error=?,finished_at=? WHERE id=?",
                (str(exc), datetime.now(timezone.utc).isoformat(), run_id)
            )
            store.commit()
            await publish(connector_id, {"event": "error", "connector_id": connector_id,
                                         "uc_id": uc_id, "error": str(exc)})
        return run_id

    async def _run_inner(self, run_id: str, connector_id: str, uc_id: str, trigger: str) -> int:
        """Returns number of new chunks upserted."""
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

        total_new_chunks = 0
        for i, sf in enumerate(files):
            raw: bytes = await loop.run_in_executor(None, connector.download, sf)
            chunks = _chunk_bytes(sf, raw)

            await publish(connector_id, {"event": "progress", "stage": "index",
                                         "connector_id": connector_id, "uc_id": uc_id,
                                         "files_total": len(files), "files_done": i,
                                         "message": f"embedding {sf.name}"})

            if chunks:
                embeddings = self._embed(uc_id, chunks)
                kv_tensors = self._kv(chunks)
                self._upsert(uc_id, chunks, embeddings, kv_tensors)
                total_new_chunks += len(chunks)

            store.execute("UPDATE sync_runs SET files_done=? WHERE id=?", (i + 1, run_id))
            store.commit()

        now = datetime.now(timezone.utc).isoformat()
        store.execute(
            "UPDATE connector_uc_scopes SET last_sync_at=?,last_delta_token=? "
            "WHERE connector_id=? AND uc_id=?",
            (now, new_token, connector_id, uc_id)
        )
        store.commit()
        return total_new_chunks

    async def _post_sync_pipeline(self, connector_id: str, uc_id: str, new_chunks: int) -> None:
        """Fire FAQ generation then KV recompute for the UC after a sync adds new chunks."""
        config_path = ROOT / "examples" / uc_id / "config.json"
        if not config_path.exists():
            return

        await publish(connector_id, {
            "event": "pipeline", "stage": "faq-gen", "uc_id": uc_id,
            "message": f"Starting FAQ generation for {new_chunks} new chunks…",
        })
        loop = asyncio.get_running_loop()

        # Step 1: FAQ generation (incremental — skips chunks already marked faq_generated_at)
        faq_ok = await loop.run_in_executor(None, _run_subprocess, [
            sys.executable, "-m", "pipeline.sleep_faq_generator",
            "--config", str(config_path),
        ])
        await publish(connector_id, {
            "event": "pipeline", "stage": "faq-gen", "uc_id": uc_id,
            "message": "FAQ generation complete" if faq_ok else "FAQ generation failed (check logs)",
            "ok": faq_ok,
        })

        # Step 2: Recompute KV for new (kv_version=null) chunks
        await publish(connector_id, {
            "event": "pipeline", "stage": "recompute-kv", "uc_id": uc_id,
            "message": "Recomputing KV tensors for new chunks…",
        })
        kv_ok = await loop.run_in_executor(None, _run_subprocess, [
            sys.executable, "-m", "pipeline.kv_indexer",
            "--config", str(config_path),
            "compute-kv", "--filter", "kv_version=null",
        ])
        await publish(connector_id, {
            "event": "pipeline", "stage": "recompute-kv", "uc_id": uc_id,
            "message": "KV recompute complete" if kv_ok else "KV recompute failed (check logs)",
            "ok": kv_ok,
        })


def _run_subprocess(cmd: list[str]) -> bool:
    """Run a pipeline subprocess synchronously; return True on success."""
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"[sync-pipeline] {cmd[2]} exited {result.returncode}: {result.stderr[-500:]}", flush=True)
        return result.returncode == 0
    except Exception as exc:
        print(f"[sync-pipeline] {cmd[2]} error: {exc}", flush=True)
        return False


def _chunk_bytes(sf: SourceFile, raw: bytes) -> list[str]:
    """Split downloaded bytes into ~512-char text chunks."""
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
    if connector_type == "wikipedia":
        from connectors.wikipedia_connector import WikipediaConnector
        return WikipediaConnector(
            topics=creds.get("topics", ""),
            language=creds.get("language", "en"),
            max_articles=int(creds.get("max_articles", 0) or 0),
        )
    if connector_type == "fda":
        from connectors.fda_connector import FDAConnector
        return FDAConnector(creds)
    if connector_type == "edgar":
        from connectors.edgar_connector import EDGARConnector
        return EDGARConnector(creds)
    if connector_type == "espn":
        from connectors.espn_connector import ESPNConnector
        return ESPNConnector(creds)
    raise ValueError(f"unknown connector type: {connector_type}")


def _make_embed_fn():
    """Return an embed function that reads each UC's config at call time."""
    _embedders: dict = {}

    def embed(uc_id: str, chunks: list[str]) -> list[list[float]]:
        if uc_id not in _embedders:
            import json
            cfg_path = ROOT / "examples" / uc_id / "config.json"
            embed_model = "BAAI/bge-small-en-v1.5"
            if cfg_path.exists():
                try:
                    raw = json.loads(cfg_path.read_text())
                    indexing = raw.get("addon_config", {}).get("indexing", raw)
                    embed_model = indexing.get("embed_model", embed_model)
                except Exception:
                    pass
            from fastembed import TextEmbedding
            _embedders[uc_id] = TextEmbedding(model_name=embed_model, show_download_progress=False)
        return [v.tolist() for v in _embedders[uc_id].embed(chunks)]

    return embed


def _make_upsert_fn():
    """Return an upsert function that writes to the UC's configured vector store."""
    import hashlib

    def upsert(uc_id: str, chunks: list[str], embeddings: list[list[float]], kv_tensors: list) -> None:
        import json, time
        from vectorstore.registry import get_store
        from vectorstore.base import Point

        cfg_path = ROOT / "examples" / uc_id / "config.json"
        if not cfg_path.exists():
            return
        try:
            raw = json.loads(cfg_path.read_text())
            if "addon_config" in raw:
                from core.config import KVForgeConfig
                dc = KVForgeConfig(**raw)
                cfg = dc.get_merged_config("indexing", "inference", "training")
                cfg.setdefault("collection", raw.get("collection", uc_id))
            else:
                cfg = raw
        except Exception as exc:
            print(f"[sync-upsert] config error for {uc_id}: {exc}", flush=True)
            return

        try:
            store_client = get_store(cfg)
            collection = cfg["collection"]
            vector_dim = cfg.get("vector_dim", 384)
            if not store_client.collection_exists(collection):
                store_client.create_collection(collection, vector_dim)

            points = []
            for chunk, vec, kv in zip(chunks, embeddings, kv_tensors):
                point_id = str(__import__("uuid").UUID(hashlib.md5(chunk.encode()).hexdigest()))
                payload = {
                    "text": chunk,
                    "page": 0,
                    "source_file": uc_id,
                    "indexed_at": int(time.time()),
                    "kv_cache": None,
                    "kv_version": None,
                    "access_count": 0,
                    "last_accessed_ts": None,
                    "avg_retrieval_rank": None,
                    "parametric_hit_count": 0,
                    "tier": "frozen",
                    "effective_from": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc).isoformat(),
                    "superseded_at": None,
                    "source_version": "",
                }
                if kv is not None:
                    import core.kv_utils as kv_utils
                    payload["kv_cache"] = kv_utils.serialize_kv(kv)
                points.append(Point(id=point_id, vector=vec, payload=payload))

            batch_size = cfg.get("upsert_batch", 128)
            for start in range(0, len(points), batch_size):
                store_client.upsert(collection, points[start:start + batch_size])
        except Exception as exc:
            print(f"[sync-upsert] upsert error for {uc_id}: {exc}", flush=True)

    return upsert


def make_default_engine() -> SyncEngine:
    """Create a SyncEngine wired to real embed, upsert, and post-sync pipeline."""
    return SyncEngine(
        connector_factory=_default_connector_factory,
        embed_fn=_make_embed_fn(),
        upsert_fn=_make_upsert_fn(),
        kv_fn=lambda chunks: [None] * len(chunks),  # KV computed by post-sync recompute step
    )
