# auth/middleware.py
import os
from datetime import datetime, timezone
from urllib.parse import quote
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse, RedirectResponse
import jwt as pyjwt
import db.store as store
from auth.models import User

SECRET = os.environ.get("KVFORGE_SECRET_KEY", "dev-secret-change-me")

_PUBLIC_PREFIXES = ("/auth/", "/webhooks/", "/static/", "/api/", "/kvq", "/ab-eval/", "/studio/", "/sync/")
_PUBLIC_EXACT = ("/",)
_API_PREFIXES = ()  # all relevant paths now covered by _PUBLIC_PREFIXES


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if path in _PUBLIC_EXACT or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)

        token = request.cookies.get("kvforge_session")
        user = _validate_token(token) if token else None

        if user is None:
            if any(path.startswith(p) for p in _API_PREFIXES):
                return JSONResponse({"detail": "not authenticated"}, status_code=401)
            return RedirectResponse(f"/auth/login?next={quote(path, safe='/')}", status_code=302)

        request.state.user = user
        return await call_next(request)


def _validate_token(token: str) -> User | None:
    try:
        payload = pyjwt.decode(token, SECRET, algorithms=["HS256"])
        row = store.fetchone(
            "SELECT user_id FROM sessions WHERE jwt_token=? AND expires_at > ?",
            (token, datetime.now(timezone.utc).isoformat())
        )
        if row is None:
            return None
        u = store.fetchone("SELECT * FROM users WHERE id=?", (payload["sub"],))
        return User.from_row(u) if u else None
    except Exception:
        return None
