"""Dashboard routes: health, status, addon listing, setup wizard, manage page."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


# ── Corpus health stubs (replaced at runtime or patched in tests) ──────────

def get_archival_candidates() -> list[dict]:
    """Return pending archival recommendations. Override in production."""
    return []


def execute_archive(chunk_id: str) -> None:
    """Execute archival for chunk_id. Override in production."""


def get_reinstatement_candidates() -> list[dict]:
    """Return reinstatement recommendations. Override in production."""
    return []


def execute_reinstate(chunk_id: str) -> None:
    """Execute reinstatement for chunk_id. Override in production."""


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

    @router.get("/api/corpus/archival-candidates")
    async def archival_candidates():
        import dashboard.routes as _self
        candidates = _self.get_archival_candidates()
        return {"candidates": candidates}

    @router.post("/api/corpus/confirm-archive")
    async def confirm_archive(body: dict):
        import dashboard.routes as _self
        chunk_id = body.get("chunk_id")
        if not chunk_id:
            raise HTTPException(status_code=400, detail="chunk_id required")
        _self.execute_archive(chunk_id)
        return {"status": "ok", "chunk_id": chunk_id}

    @router.get("/api/corpus/reinstatement-candidates")
    async def reinstatement_candidates():
        import dashboard.routes as _self
        candidates = _self.get_reinstatement_candidates()
        return {"candidates": candidates}

    @router.post("/api/corpus/confirm-reinstate")
    async def confirm_reinstate(body: dict):
        import dashboard.routes as _self
        chunk_id = body.get("chunk_id")
        if not chunk_id:
            raise HTTPException(status_code=400, detail="chunk_id required")
        _self.execute_reinstate(chunk_id)
        return {"status": "ok", "chunk_id": chunk_id}

    return router
