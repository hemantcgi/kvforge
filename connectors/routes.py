# connectors/routes.py
import asyncio
import db.store as store
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from connectors.registry import ConnectorRegistry

# webhook_secret is stored in plaintext (not Fernet-encrypted) because webhook
# verification requires the raw HMAC secret for signature comparison at runtime.
connector_router = APIRouter(prefix="/studio/api/connectors", tags=["connectors"])
_registry = ConnectorRegistry()

_ADMIN_ONLY = ("admin",)
_EDITOR_UP = ("admin", "editor")
_ANY_AUTH = ("admin", "editor", "viewer")


def _require_role(request: Request, roles: tuple) -> JSONResponse | None:
    u = getattr(request.state, "user", None)
    if not u or u.role not in roles:
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    return None


@connector_router.get("")
async def list_connectors(request: Request):
    if err := _require_role(request, _ANY_AUTH):
        return err
    return _registry.list_all()


@connector_router.post("")
async def create_connector(request: Request):
    if err := _require_role(request, _ADMIN_ONLY):
        return err
    body = await request.json()
    missing = [f for f in ("type", "name") if f not in body]
    if missing:
        return JSONResponse({"detail": f"missing required fields: {missing}"}, status_code=400)
    valid_types = ("gdrive", "s3", "sharepoint", "wikipedia", "fda", "edgar", "espn")
    if body["type"] not in valid_types:
        return JSONResponse({"detail": f"type must be one of {valid_types}"}, status_code=400)
    cfg = _registry.create(
        connector_type=body["type"],
        name=body["name"],
        credentials=body.get("credentials", {}),
        created_by=request.state.user.id,
        schedule_cron=body.get("schedule_cron"),
        webhook_secret=body.get("webhook_secret"),
    )
    return cfg


@connector_router.put("/{cid}")
async def update_connector(cid: str, request: Request):
    if err := _require_role(request, _ADMIN_ONLY):
        return err
    body = await request.json()
    try:
        kwargs: dict = {}
        if "credentials" in body:
            kwargs["credentials"] = body["credentials"]
        if "schedule_cron" in body:
            kwargs["schedule_cron"] = body["schedule_cron"]
        if "webhook_secret" in body:
            kwargs["webhook_secret"] = body["webhook_secret"]
        if "name" in body:
            kwargs["name"] = body["name"]
        return _registry.update(cid, **kwargs)
    except KeyError:
        return JSONResponse({"detail": "not found"}, status_code=404)


@connector_router.delete("/{cid}")
async def delete_connector(cid: str, request: Request):
    if err := _require_role(request, _ADMIN_ONLY):
        return err
    if _registry.get(cid) is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    _registry.delete(cid)
    return {"ok": True}


@connector_router.post("/{cid}/test")
async def test_connector(cid: str, request: Request):
    if err := _require_role(request, _ADMIN_ONLY):
        return err
    try:
        creds = _registry.get_credentials(cid)
        cfg = _registry.get(cid)
        if cfg is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        result = await asyncio.wait_for(_run_test(cfg["type"], creds), timeout=10.0)
        return result
    except asyncio.TimeoutError:
        return JSONResponse({"ok": False, "error": "timeout after 10s"}, status_code=200)
    except Exception:
        return {"ok": False, "error": "connectivity check failed"}


async def _run_test(connector_type: str, creds: dict) -> dict:
    if connector_type == "gdrive":
        try:
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build
        except ImportError:
            return {"ok": False, "error": "google-auth is not installed on this server"}
        import json
        info = json.loads(creds.get("service_account_json", "{}"))
        sa_creds = Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive.readonly"])
        svc = build("drive", "v3", credentials=sa_creds, cache_discovery=False)
        files = svc.files().list(pageSize=1).execute()
        return {"ok": True, "detail": f"Connected — {len(files.get('files', []))} files visible"}

    elif connector_type == "s3":
        try:
            import boto3
        except ImportError:
            return {"ok": False, "error": "boto3 is not installed on this server"}
        s3 = boto3.client(
            "s3",
            aws_access_key_id=creds.get("access_key_id"),
            aws_secret_access_key=creds.get("secret_access_key"),
            region_name=creds.get("region", "us-east-1"),
        )
        s3.head_bucket(Bucket=creds.get("bucket", ""))
        return {"ok": True, "detail": "S3 bucket reachable"}

    elif connector_type == "sharepoint":
        try:
            import msal
            import httpx
        except ImportError:
            return {"ok": False, "error": "msal or httpx is not installed on this server"}
        msal_app = msal.ConfidentialClientApplication(
            creds["client_id"],
            authority=f"https://login.microsoftonline.com/{creds['tenant_id']}",
            client_credential=creds["client_secret"],
        )
        result = msal_app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"])
        if "error" in result:
            return {"ok": False, "error": result.get("error_description", result["error"])}
        async with httpx.AsyncClient() as hc:
            r = await hc.get(
                f"https://graph.microsoft.com/v1.0/sites/{creds.get('site_url', '')}",
                headers={"Authorization": f"Bearer {result['access_token']}"},
            )
        return {"ok": r.status_code == 200, "detail": f"HTTP {r.status_code}"}

    elif connector_type == "wikipedia":
        import httpx, urllib.parse
        raw = creds.get("topics", "").split(",")[0].strip()
        topic = raw or "Python_(programming_language)"
        encoded = urllib.parse.quote(topic, safe="()")
        r = httpx.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}",
            timeout=8,
            headers={"User-Agent": "KVForge/2.1 (https://github.com/flotorch/kvforge; contact@flotorch.ai)"},
        )
        if r.status_code == 200:
            return {"ok": True, "detail": f"Wikipedia reachable — '{topic}' found"}
        if r.status_code == 404:
            return {"ok": False, "error": f"Article '{topic}' not found on Wikipedia — check the title spelling"}
        return {"ok": False, "error": f"Wikipedia returned HTTP {r.status_code}"}

    elif connector_type == "fda":
        import httpx
        drug = creds.get("drug_name", "aspirin")
        r = httpx.get(
            "https://api.fda.gov/drug/label.json",
            params={"search": f'openfda.brand_name:"{drug}"', "limit": 1},
            timeout=10,
        )
        if r.status_code in (200, 404):
            count = len(r.json().get("results", [])) if r.status_code == 200 else 0
            return {"ok": True, "detail": f"openFDA reachable — {count} labels for '{drug}'"}
        return {"ok": False, "error": f"openFDA returned HTTP {r.status_code}"}

    elif connector_type == "edgar":
        import httpx
        ticker = creds.get("ticker", "AAPL")
        r = httpx.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": f'"{ticker}"', "forms": "10-K", "dateRange": "custom",
                    "startdt": "2020-01-01"},
            headers={"User-Agent": "KVForge research@kvforge.ai"},
            timeout=10,
        )
        if r.status_code == 200:
            count = len(r.json().get("hits", {}).get("hits", []))
            return {"ok": True, "detail": f"EDGAR reachable — {count} filings for '{ticker}'"}
        return {"ok": False, "error": f"EDGAR returned HTTP {r.status_code}"}

    elif connector_type == "espn":
        import httpx
        sport = creds.get("sport", "football")
        league = creds.get("league", "nfl")
        r = httpx.get(
            f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/news",
            params={"limit": 1},
            timeout=8,
        )
        if r.status_code == 200:
            count = len(r.json().get("articles", []))
            return {"ok": True, "detail": f"ESPN reachable — {count} articles for {sport}/{league}"}
        return {"ok": False, "error": f"ESPN returned HTTP {r.status_code}"}

    return {"ok": False, "error": f"unknown connector type: {connector_type}"}


@connector_router.get("/{cid}/scopes")
async def list_scopes(cid: str, request: Request):
    if err := _require_role(request, _ANY_AUTH):
        return err
    return _registry.list_scopes(cid)


@connector_router.post("/{cid}/scopes")
async def add_scope(cid: str, request: Request):
    if err := _require_role(request, _EDITOR_UP):
        return err
    body = await request.json()
    if "uc_id" not in body:
        return JSONResponse({"detail": "missing required field: uc_id"}, status_code=400)
    if _registry.get(cid) is None:
        return JSONResponse({"detail": "connector not found"}, status_code=404)
    _registry.upsert_scope(cid, body["uc_id"], body.get("scope_config", {}))
    return {"ok": True}


@connector_router.delete("/{cid}/scopes/{uc_id}")
async def delete_scope(cid: str, uc_id: str, request: Request):
    if err := _require_role(request, _EDITOR_UP):
        return err
    _registry.delete_scope(cid, uc_id)
    return {"ok": True}


@connector_router.post("/{cid}/sync")
async def trigger_sync(cid: str, request: Request):
    if err := _require_role(request, _EDITOR_UP):
        return err
    scopes = _registry.list_scopes(cid)
    if not scopes:
        return JSONResponse({"detail": "no scopes configured for this connector"}, status_code=400)
    from connectors.sync_engine import make_default_engine
    import asyncio
    engine = make_default_engine()
    for scope in scopes:
        asyncio.create_task(engine.run(cid, scope["uc_id"], "manual"))
    return {"ok": True, "triggered_scopes": [s["uc_id"] for s in scopes]}


# Sync-runs router — included separately in portal
_sync_runs_router = APIRouter(prefix="/studio/api", tags=["sync-runs"])


@_sync_runs_router.get("/sync-runs")
async def list_sync_runs(request: Request):
    u = getattr(request.state, "user", None)
    if not u:
        return JSONResponse({"detail": "not authenticated"}, status_code=401)
    rows = store.fetchall(
        "SELECT sr.*, cc.name as connector_name FROM sync_runs sr "
        "LEFT JOIN connector_configs cc ON sr.connector_id=cc.id "
        "ORDER BY sr.started_at DESC LIMIT 50"
    )
    return [dict(r) for r in rows]
