# studio/routes.py
"""FastAPI router — aggregates api_router from api.py and will add SSE + page routes in Task 6."""

from pathlib import Path
from fastapi import APIRouter

from studio.api import api_router

ROOT = Path(__file__).resolve().parent.parent

router = APIRouter()
router.include_router(api_router)
