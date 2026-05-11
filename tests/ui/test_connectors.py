# tests/ui/test_connectors.py
"""Playwright tests for connector management UI."""
import json
import uuid
import time
import httpx
import pytest


def _get_tok(page) -> str:
    cookies = page.context.cookies()
    for c in cookies:
        if c["name"] == "kvforge_session":
            return c["value"]
    return ""


def _api(base_url: str, tok: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, cookies={"kvforge_session": tok}, timeout=10)


def test_connectors_page_loads(admin_page, app_server):
    base_url, _ = app_server
    admin_page.goto(f"{base_url}/studio/connectors")
    admin_page.wait_for_load_state("networkidle")
    assert "Connectors" in admin_page.content()
    assert admin_page.locator("#conn-list").count() == 1


def test_add_gdrive_connector_via_api(admin_page, app_server):
    base_url, _ = app_server
    tok = _get_tok(admin_page)
    with _api(base_url, tok) as client:
        r = client.post("/studio/api/connectors", json={
            "type": "gdrive",
            "name": f"Test GDrive {uuid.uuid4()}",
            "credentials": {"service_account_json": json.dumps({"type": "service_account"})},
        })
    assert r.status_code == 200
    assert r.json()["credentials"] == "●●●●●●"


def test_credentials_masked_in_api_response(admin_page, app_server):
    base_url, _ = app_server
    tok = _get_tok(admin_page)
    secret_val = f"sUp3r5ecr3t-{uuid.uuid4()}"
    conn_name = f"Masked S3 {uuid.uuid4()}"
    with _api(base_url, tok) as client:
        client.post("/studio/api/connectors", json={
            "type": "s3", "name": conn_name,
            "credentials": {"access_key_id": "AKIAIOSFODNN7", "secret": secret_val},
        })
        r = client.get("/studio/api/connectors")
    body = r.text
    # Connector must appear in the list (proves the masked response is from the real data path)
    assert conn_name in body
    assert secret_val not in body
    assert "AKIAIOSFODNN7" not in body


def test_add_uc_scope(admin_page, app_server):
    base_url, _ = app_server
    tok = _get_tok(admin_page)
    with _api(base_url, tok) as client:
        r = client.post("/studio/api/connectors", json={
            "type": "gdrive", "name": f"Scope Test {uuid.uuid4()}",
            "credentials": {"service_account_json": "{}"},
        })
        cid = r.json()["id"]
        r2 = client.post(f"/studio/api/connectors/{cid}/scopes", json={
            "uc_id": "uc1",
            "scope_config": {"folder_id": "drive-folder-xyz"},
        })
    assert r2.status_code == 200
    with _api(base_url, tok) as client:
        scopes = client.get(f"/studio/api/connectors/{cid}/scopes").json()
    assert len(scopes) == 1
    assert scopes[0]["scope_config"]["folder_id"] == "drive-folder-xyz"


def test_viewer_cannot_create_connector(viewer_page, app_server):
    base_url, _ = app_server
    tok = _get_tok(viewer_page)
    with _api(base_url, tok) as client:
        r = client.post("/studio/api/connectors", json={
            "type": "s3", "name": "Viewer attempt",
            "credentials": {},
        })
    assert r.status_code == 403


def test_manual_sync_trigger(admin_page, app_server):
    """Sync trigger returns 200 ok=True — actual sync may fail on server (no real creds)."""
    base_url, _ = app_server
    tok = _get_tok(admin_page)
    with _api(base_url, tok) as client:
        r = client.post("/studio/api/connectors", json={
            "type": "gdrive", "name": f"Sync Test {uuid.uuid4()}",
            "credentials": {"service_account_json": "{}"},
        })
        cid = r.json()["id"]
        client.post(f"/studio/api/connectors/{cid}/scopes",
                    json={"uc_id": "uc-sync", "scope_config": {"folder_id": "f1"}})
        r2 = client.post(f"/studio/api/connectors/{cid}/sync")
    assert r2.status_code == 200
    assert r2.json()["ok"] is True


def test_sync_history_recorded(admin_page, app_server):
    """After triggering sync, a sync_runs record is created (even if it errors)."""
    base_url, _ = app_server
    tok = _get_tok(admin_page)
    with _api(base_url, tok) as client:
        r = client.post("/studio/api/connectors", json={
            "type": "s3", "name": f"History S3 {uuid.uuid4()}",
            "credentials": {"access_key_id": "A", "secret": "S", "bucket": "b"},
        })
        cid = r.json()["id"]
        client.post(f"/studio/api/connectors/{cid}/scopes",
                    json={"uc_id": "uc-hist", "scope_config": {"bucket": "b", "prefix": ""}})
        client.post(f"/studio/api/connectors/{cid}/sync")
        time.sleep(1.5)
        runs = client.get("/studio/api/sync-runs").json()
    assert any(run["connector_id"] == cid for run in runs)


def test_delete_connector(admin_page, app_server):
    base_url, _ = app_server
    tok = _get_tok(admin_page)
    with _api(base_url, tok) as client:
        r = client.post("/studio/api/connectors", json={
            "type": "s3", "name": f"To Delete {uuid.uuid4()}",
            "credentials": {"k": "v"},
        })
        cid = r.json()["id"]
        r2 = client.delete(f"/studio/api/connectors/{cid}")
    assert r2.status_code == 200
    with _api(base_url, tok) as client:
        all_conn = client.get("/studio/api/connectors").json()
    assert not any(c["id"] == cid for c in all_conn)
