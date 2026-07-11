"""Per-use-case KVForge Dashboard — setup wizard + addon management.

Usage::

    uvicorn dashboard.app:create_app --factory -- --config myconfig.json
    # or via kvforge CLI:
    kvforge start --config myconfig.json
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from dashboard.routes import make_router

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "dashboard"


def create_app(config_path: str = "config.json") -> FastAPI:
    """Create and return the per-UC dashboard FastAPI application.

    Args:
        config_path: Path to the KVForgeConfig JSON file.
            If the file does not exist, the dashboard shows the setup wizard.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="KVForge Dashboard",
        description="Per-use-case KVForge setup and addon management",
        version="1.0.0",
    )

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    router = make_router(config_path=config_path, templates=templates)
    app.include_router(router)

    return app
