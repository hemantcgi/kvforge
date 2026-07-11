"""
Centralized activity log for KVForge Studio.
All studio events are appended as JSONL to ~/.kvforge/studio_activity.jsonl.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_LOG_FILE = Path.home() / ".kvforge" / "studio_activity.jsonl"
_lock = threading.Lock()

CATEGORIES = ("pipeline", "resource", "auth", "system", "connector")
SEVERITIES = ("info", "success", "warning", "error")

_CAT_LABELS = {
    "pipeline":  "Pipeline",
    "resource":  "Resource",
    "auth":      "Auth",
    "system":    "System",
    "connector": "Connector",
}


def log_event(
    category: str,
    action: str,
    message: str,
    details: dict | None = None,
    uc_id: str | None = None,
    severity: str = "info",
) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "category": category,
        "action": action,
        "severity": severity,
        "message": message,
        "uc_id": uc_id or "",
        "details": details or {},
    }
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with _LOG_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def query_logs(
    categories: list[str] | None = None,
    severities: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    search: str | None = None,
    uc_id: str | None = None,
    limit: int = 500,
) -> list[dict]:
    if not _LOG_FILE.exists():
        return []
    try:
        lines = _LOG_FILE.read_text().splitlines()
    except Exception:
        return []

    results = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if categories and entry.get("category") not in categories:
            continue
        if severities and entry.get("severity") not in severities:
            continue
        ts = entry.get("ts", "")
        if since and ts < since:
            continue
        if until and ts > until + "T99":
            continue
        if uc_id and entry.get("uc_id") != uc_id:
            continue
        if search:
            hay = (entry.get("message", "") + " " + entry.get("action", "")).lower()
            if search.lower() not in hay:
                continue
        results.append(entry)
        if len(results) >= limit:
            break
    return results


def get_stats() -> dict:
    if not _LOG_FILE.exists():
        return {"total": 0, "by_category": {}, "by_severity": {}, "by_day": {}}
    try:
        lines = _LOG_FILE.read_text().splitlines()
    except Exception:
        return {"total": 0, "by_category": {}, "by_severity": {}, "by_day": {}}

    total = 0
    by_category: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    by_day: dict[str, int] = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        total += 1
        cat = entry.get("category", "system")
        sev = entry.get("severity", "info")
        day = entry.get("ts", "")[:10]
        by_category[cat] = by_category.get(cat, 0) + 1
        by_severity[sev] = by_severity.get(sev, 0) + 1
        if day:
            by_day[day] = by_day.get(day, 0) + 1

    return {"total": total, "by_category": by_category, "by_severity": by_severity, "by_day": by_day}
