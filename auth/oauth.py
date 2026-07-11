# auth/oauth.py
"""OAuth2 / OIDC integration. Each provider is only active if its env vars are set."""
import os, uuid
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
import db.store as store
from auth.models import User
from auth.routes import _create_session

router = APIRouter(prefix="/auth/oauth", tags=["oauth"])

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
MS_CLIENT_ID = os.environ.get("MICROSOFT_CLIENT_ID", "")
MS_CLIENT_SECRET = os.environ.get("MICROSOFT_CLIENT_SECRET", "")
MS_TENANT_ID = os.environ.get("MICROSOFT_TENANT_ID", "common")
AWS_POOL_ID = os.environ.get("AWS_COGNITO_POOL_ID", "")
AWS_CLIENT_ID = os.environ.get("AWS_COGNITO_CLIENT_ID", "")
AWS_REGION = os.environ.get("AWS_COGNITO_REGION", "us-east-1")


def _upsert_oauth_user(email: str, provider: str, provider_id: str) -> User:
    """Find or create a user for an OAuth login. First-ever user gets admin."""
    from auth.routes import _count_users
    row = store.fetchone("SELECT * FROM users WHERE provider=? AND provider_id=?",
                         (provider, provider_id))
    if row:
        return User.from_row(row)
    # Check by email (invite may have pre-created them as local user)
    row = store.fetchone("SELECT * FROM users WHERE email=?", (email,))
    if row:
        store.execute("UPDATE users SET provider=?, provider_id=? WHERE id=?",
                      (provider, provider_id, row["id"]))
        store.commit()
        return User.from_row(store.fetchone("SELECT * FROM users WHERE id=?", (row["id"],)))
    role = "admin" if _count_users() == 0 else "viewer"
    uid = str(uuid.uuid4())
    store.execute(
        "INSERT INTO users(id,email,role,provider,provider_id) VALUES(?,?,?,?,?)",
        (uid, email, role, provider, provider_id)
    )
    store.commit()
    return User.from_row(store.fetchone("SELECT * FROM users WHERE id=?", (uid,)))


def _oauth_redirect(request: Request, user: User) -> RedirectResponse:
    tok, _ = _create_session(user)
    resp = RedirectResponse("/studio/", status_code=302)
    resp.set_cookie("kvforge_session", tok, httponly=True, secure=False, samesite="lax")
    return resp


# ── Google ────────────────────────────────────────────────────────────────────

@router.get("/google")
async def google_login(request: Request):
    if not GOOGLE_CLIENT_ID:
        return RedirectResponse("/auth/login?error=Google+OAuth+not+configured")
    from authlib.integrations.starlette_client import OAuth
    oauth = OAuth()
    oauth.register("google",
        client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    redirect_uri = str(request.url_for("google_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback", name="google_callback")
async def google_callback(request: Request):
    if not GOOGLE_CLIENT_ID:
        return RedirectResponse("/auth/login?error=Google+OAuth+not+configured")
    from authlib.integrations.starlette_client import OAuth
    oauth = OAuth()
    oauth.register("google",
        client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    try:
        token = await oauth.google.authorize_access_token(request)
        userinfo = token.get("userinfo") or await oauth.google.userinfo(token=token)
        user = _upsert_oauth_user(userinfo["email"], "google", userinfo["sub"])
        return _oauth_redirect(request, user)
    except Exception as e:
        return RedirectResponse(f"/auth/login?error=Google+login+failed")


# ── Microsoft ─────────────────────────────────────────────────────────────────

@router.get("/microsoft")
async def microsoft_login(request: Request):
    if not MS_CLIENT_ID:
        return RedirectResponse("/auth/login?error=Microsoft+OAuth+not+configured")
    try:
        import msal
        msal_app = msal.ConfidentialClientApplication(
            MS_CLIENT_ID,
            authority=f"https://login.microsoftonline.com/{MS_TENANT_ID}",
            client_credential=MS_CLIENT_SECRET,
        )
        redirect_uri = str(request.url_for("microsoft_callback"))
        flow = msal_app.initiate_auth_code_flow(
            ["openid", "email", "profile"], redirect_uri=redirect_uri)
        request.session["msal_flow"] = flow
        return RedirectResponse(flow["auth_uri"])
    except Exception:
        return RedirectResponse("/auth/login?error=Microsoft+OAuth+not+configured")


@router.get("/microsoft/callback", name="microsoft_callback")
async def microsoft_callback(request: Request):
    if not MS_CLIENT_ID:
        return RedirectResponse("/auth/login?error=Microsoft+OAuth+not+configured")
    try:
        import msal
        flow = request.session.pop("msal_flow", {})
        msal_app = msal.ConfidentialClientApplication(
            MS_CLIENT_ID,
            authority=f"https://login.microsoftonline.com/{MS_TENANT_ID}",
            client_credential=MS_CLIENT_SECRET,
        )
        result = msal_app.acquire_token_by_auth_code_flow(flow, dict(request.query_params))
        if "error" in result:
            return RedirectResponse(f"/auth/login?error=Microsoft+login+failed")
        claims = result.get("id_token_claims", {})
        user = _upsert_oauth_user(claims.get("email", ""), "microsoft", claims.get("oid", ""))
        return _oauth_redirect(request, user)
    except Exception:
        return RedirectResponse("/auth/login?error=Microsoft+login+failed")


# ── AWS Cognito ───────────────────────────────────────────────────────────────

@router.get("/aws")
async def aws_login(request: Request):
    if not AWS_CLIENT_ID:
        return RedirectResponse("/auth/login?error=AWS+OAuth+not+configured")
    try:
        from authlib.integrations.starlette_client import OAuth
        oauth = OAuth()
        domain = f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{AWS_POOL_ID}"
        oauth.register("aws",
            client_id=AWS_CLIENT_ID,
            server_metadata_url=f"{domain}/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
        redirect_uri = str(request.url_for("aws_callback"))
        return await oauth.aws.authorize_redirect(request, redirect_uri)
    except Exception:
        return RedirectResponse("/auth/login?error=AWS+OAuth+not+configured")


@router.get("/aws/callback", name="aws_callback")
async def aws_callback(request: Request):
    if not AWS_CLIENT_ID:
        return RedirectResponse("/auth/login?error=AWS+OAuth+not+configured")
    try:
        from authlib.integrations.starlette_client import OAuth
        oauth = OAuth()
        domain = f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{AWS_POOL_ID}"
        oauth.register("aws",
            client_id=AWS_CLIENT_ID,
            server_metadata_url=f"{domain}/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
        token = await oauth.aws.authorize_access_token(request)
        userinfo = token.get("userinfo") or await oauth.aws.userinfo(token=token)
        user = _upsert_oauth_user(userinfo["email"], "aws", userinfo["sub"])
        return _oauth_redirect(request, user)
    except Exception:
        return RedirectResponse("/auth/login?error=AWS+login+failed")
