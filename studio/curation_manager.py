# studio/curation_manager.py
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURATED_FILENAME = "faqs_curated.json"


def _path(uc_id: str) -> Path:
    p = (ROOT / "examples" / uc_id / CURATED_FILENAME).resolve()
    allowed = (ROOT / "examples").resolve()
    if not p.is_relative_to(allowed):
        raise ValueError(f"Invalid uc_id escapes examples directory: {uc_id!r}")
    return p


def _load(uc_id: str) -> list:
    p = _path(uc_id)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write(uc_id: str, records: list) -> None:
    p = _path(uc_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(records, indent=2))
        os.replace(tmp, p)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def append(uc_id: str, question: str, answer: str, source_model: str = "model_b") -> dict:
    records = _load(uc_id)
    records.append({
        "question": question,
        "answer": answer,
        "source_model": source_model,
        "curated_at": datetime.now(timezone.utc).isoformat(),
    })
    _write(uc_id, records)
    return get_status(uc_id)


def get_status(uc_id: str) -> dict:
    from studio.settings_manager import get_setting
    records = _load(uc_id)
    count = len(records)
    threshold = int(get_setting("curation_threshold") or 50)
    return {
        "count": count,
        "threshold": threshold,
        "pct": round((count / threshold) * 100, 1) if threshold else 0.0,
        "at_threshold": count >= threshold,
    }


def get_samples(uc_id: str, n: int = 5) -> list:
    records = _load(uc_id)
    return records[-n:] if len(records) > n else records
