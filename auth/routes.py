# auth/routes.py
import html as _html
import os, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import bcrypt
import jwt as pyjwt
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import db.store as store
from auth.models import User

SECRET = os.environ.get("KVFORGE_SECRET_KEY", "dev-secret-change-me")
TEMPLATES = Path(__file__).resolve().parent.parent / "templates" / "studio" / "auth"
router = APIRouter(prefix="/auth", tags=["auth"])


def _make_jwt(uid: str, role: str) -> tuple[str, datetime]:
    exp = datetime.now(timezone.utc) + timedelta(days=7)
    tok = pyjwt.encode({"sub": uid, "role": role, "exp": exp}, SECRET, algorithm="HS256")
    return tok, exp


def _create_session(user: User) -> tuple[str, datetime]:
    tok, exp = _make_jwt(user.id, user.role)
    store.execute(
        "INSERT INTO sessions(id,user_id,jwt_token,expires_at) VALUES(?,?,?,?)",
        (str(uuid.uuid4()), user.id, tok, exp.isoformat())
    )
    store.commit()
    return tok, exp


def _count_users() -> int:
    row = store.fetchone("SELECT COUNT(*) as n FROM users")
    return row["n"] if row else 0


def _render(name: str, **ctx) -> str:
    html = (TEMPLATES / name).read_text()
    for k, v in ctx.items():
        html = html.replace("{{" + k + "}}", str(v) if v else "")
    return html


@router.get("/login", response_class=HTMLResponse)
async def login_page(error: str = ""):
    env = os.environ
    err_html = f'<div class="err">{_html.escape(error)}</div>' if error else ""
    google_btn = '<a class="oauth-btn" href="/auth/oauth/google">Sign in with Google</a>' if env.get("GOOGLE_CLIENT_ID") else ""
    ms_btn = '<a class="oauth-btn" href="/auth/oauth/microsoft">Sign in with Microsoft</a>' if env.get("MICROSOFT_CLIENT_ID") else ""
    aws_btn = '<a class="oauth-btn" href="/auth/oauth/aws">Sign in with AWS</a>' if env.get("AWS_COGNITO_CLIENT_ID") else ""
    return HTMLResponse(_render("login.html", error=err_html, google_btn=google_btn, microsoft_btn=ms_btn, aws_btn=aws_btn))


@router.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    row = store.fetchone("SELECT * FROM users WHERE email=? AND provider='local'", (email,))
    if not row or not bcrypt.checkpw(password.encode(), row["hashed_pw"].encode()):
        return RedirectResponse("/auth/login?error=Invalid+credentials", status_code=302)
    user = User.from_row(row)
    tok, exp = _create_session(user)
    next_url = request.query_params.get("next", "/studio/")
    resp = RedirectResponse(next_url, status_code=302)
    resp.set_cookie("kvforge_session", tok, httponly=True, secure=False, samesite="lax")
    return resp


@router.get("/logout")
async def logout(request: Request):
    tok = request.cookies.get("kvforge_session")
    if tok:
        store.execute("DELETE FROM sessions WHERE jwt_token=?", (tok,))
        store.commit()
    resp = RedirectResponse("/auth/login", status_code=302)
    resp.delete_cookie("kvforge_session")
    return resp


@router.get("/me")
async def me(request: Request):
    u = getattr(request.state, "user", None)
    if not u:
        return JSONResponse({"detail": "not authenticated"}, status_code=401)
    return {"id": u.id, "email": u.email, "role": u.role, "provider": u.provider}


@router.post("/invite")
async def invite(request: Request):
    u = getattr(request.state, "user", None)
    if not u or u.role != "admin":
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    body = await request.json()
    email, role = body["email"], body.get("role", "viewer")
    tok = str(uuid.uuid4())
    exp = datetime.now(timezone.utc) + timedelta(hours=48)
    store.execute(
        "INSERT INTO invite_tokens(token,email,role,created_by,expires_at) VALUES(?,?,?,?,?)",
        (tok, email, role, u.id, exp.isoformat())
    )
    store.commit()
    base = str(request.base_url).rstrip("/")
    return {"signup_url": f"{base}/auth/signup?token={tok}", "expires_at": exp.isoformat()}


@router.get("/signup", response_class=HTMLResponse)
async def signup_page(token: str = ""):
    now = datetime.now(timezone.utc).isoformat()
    inv = store.fetchone(
        "SELECT * FROM invite_tokens WHERE token=? AND used_at IS NULL AND expires_at > ?",
        (token, now)
    ) if token else None
    if not inv:
        return HTMLResponse("<h1 style='color:#ce9178;font-family:sans-serif;padding:40px'>Invalid or expired invite link</h1>", status_code=400)
    return HTMLResponse(_render("signup.html",
        email=_html.escape(inv["email"]),
        role=_html.escape(inv["role"]),
        token=_html.escape(token)
    ))


@router.post("/signup")
async def signup(token: str = Form(...), password: str = Form(...)):
    now = datetime.now(timezone.utc).isoformat()
    inv = store.fetchone(
        "SELECT * FROM invite_tokens WHERE token=? AND used_at IS NULL AND expires_at > ?",
        (token, now)
    )
    if not inv:
        return JSONResponse({"detail": "invalid or expired token"}, status_code=400)
    role = "admin" if _count_users() == 0 else inv["role"]
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()
    uid = str(uuid.uuid4())
    store.execute(
        "INSERT INTO users(id,email,hashed_pw,role,provider,invited_by) VALUES(?,?,?,?,?,?)",
        (uid, inv["email"], hashed, role, "local", inv["created_by"])
    )
    store.execute("UPDATE invite_tokens SET used_at=? WHERE token=?", (now, token))
    store.commit()
    user_row = store.fetchone("SELECT * FROM users WHERE id=?", (uid,))
    tok, _ = _create_session(User.from_row(user_row))
    resp = RedirectResponse("/studio/", status_code=302)
    resp.set_cookie("kvforge_session", tok, httponly=True, secure=False, samesite="lax")
    return resp


@router.get("/studio-users", tags=["admin"])
async def list_users(request: Request):
    u = getattr(request.state, "user", None)
    if not u or u.role != "admin":
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    rows = store.fetchall("SELECT id,email,role,provider,created_at FROM users ORDER BY created_at")
    return [dict(r) for r in rows]


@router.put("/studio-users/{uid}/role", tags=["admin"])
async def change_role(uid: str, request: Request):
    u = getattr(request.state, "user", None)
    if not u or u.role != "admin":
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    body = await request.json()
    new_role = body.get("role")
    if new_role not in ("admin", "editor", "viewer"):
        return JSONResponse({"detail": "invalid role"}, status_code=422)
    store.execute("UPDATE users SET role=? WHERE id=?", (new_role, uid))
    store.commit()
    return {"ok": True}
