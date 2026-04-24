"""FastAPI router for cross-UC Flywheel summary in KVForge Studio."""
from fastapi import APIRouter
from pathlib import Path
import json

import core.analytics as _analytics

flywheel_router = APIRouter()

ROOT = Path(__file__).resolve().parent.parent


def _get_all_uc_configs():
    """Return list of objects with .id and .config (dict) for all known UCs.

    Reads datasource_*.json config files from the project root.
    """
    class _UC:
        def __init__(self, uc_id, config):
            self.id = uc_id
            self.config = config

    ucs = []
    for cfg_file in sorted(ROOT.glob("datasource_*.json")):
        try:
            with open(cfg_file) as f:
                data = json.load(f)
            uc_id = cfg_file.stem.replace("datasource_", "")
            ucs.append(_UC(uc_id, data))
        except Exception:
            continue
    return ucs


@flywheel_router.get("/api/flywheel/all")
def get_all_flywheel_summaries():
    """Return flywheel summary for every known UC — powers Studio cross-UC panel."""
    ucs = _get_all_uc_configs()
    results = []
    for uc in ucs:
        try:
            _analytics.init_db(uc.config)
            summary = _analytics.get_flywheel_summary(uc.config)
            experiments = _analytics.get_modelscout_experiments(uc.config)
            results.append({
                "uc_id": uc.id,
                "summary": summary,
                "has_experiments": len(experiments) > 0,
            })
        except Exception:
            results.append({"uc_id": uc.id, "summary": {"no_data": True},
                            "has_experiments": False})
    return results
