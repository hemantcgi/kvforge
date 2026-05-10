"""Dashboard routes: health, status, addon listing, setup wizard, manage page."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


def make_router(config_path: str, templates: Jinja2Templates) -> APIRouter:
    router = APIRouter()
    cfg_path = Path(config_path)

    def _load_cfg():
        """Return KVForgeConfig or None if config file does not exist."""
        if not cfg_path.exists():
            return None
        from core.config import load_config
        return load_config(str(cfg_path))

    @router.get("/api/health")
    def health():
        return {"status": "ok", "service": "kvforge-dashboard"}

    @router.get("/api/status")
    def status():
        cfg = _load_cfg()
        if cfg is None:
            return {"configured": False}
        return {
            "configured": True,
            "use_case_name": cfg.use_case_name,
            "collection": cfg.collection,
            "version_file": cfg.version_file,
            "addons": cfg.addons,
        }

    @router.get("/api/addons")
    def list_addons():
        from addons.registry import AddonRegistry
        AddonRegistry.load_builtins()
        cfg = _load_cfg()
        active_addons = set(cfg.addons) if cfg else set()
        return {
            "available": [
                {
                    "name": m.name,
                    "display_name": m.display_name,
                    "description": m.description,
                    "requires": m.requires,
                    "active": m.name in active_addons,
                }
                for m in AddonRegistry.all_available()
            ]
        }

    @router.get("/", response_class=HTMLResponse)
    def root(request: Request):
        cfg = _load_cfg()
        if cfg is None:
            return templates.TemplateResponse(
                request, "setup.html",
                {"config_path": str(cfg_path)},
            )
        from addons.registry import AddonRegistry
        AddonRegistry.load_builtins()
        active = set(cfg.addons)
        addons_data = [
            {
                "name": m.name,
                "display_name": m.display_name,
                "description": m.description,
                "requires": m.requires,
                "active": m.name in active,
            }
            for m in AddonRegistry.all_available()
        ]
        return templates.TemplateResponse(
            request, "manage.html",
            {
                "use_case_name": cfg.use_case_name,
                "collection": cfg.collection,
                "addons": addons_data,
                "config_path": str(cfg_path),
            },
        )

    return router
