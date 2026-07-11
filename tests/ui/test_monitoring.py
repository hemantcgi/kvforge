# tests/ui/test_monitoring.py
"""Playwright tests for monitoring panel and error connector display."""
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


def test_studio_hub_loads(admin_page, app_server):
    """Studio hub page loads without 500."""
    base_url, _ = app_server
    response = admin_page.goto(f"{base_url}/studio/")
    admin_page.wait_for_load_state("networkidle")
    assert response.status != 500
    assert "Internal Server Error" not in admin_page.content()
    assert admin_page.locator("body").count() == 1


def test_uc_detail_page_loads(admin_page, app_server):
    """UC detail page loads for a known UC ID (or gracefully 404s, not 500)."""
    base_url, _ = app_server
    # Try the first UC from the registry API
    tok = _get_tok(admin_page)
    uc_id = "uc1"  # default from USE_CASES in kvforge_portal.py
    with httpx.Client(base_url=base_url, cookies={"kvforge_session": tok}) as client:
        r = client.get("/studio/api/registry")
        if r.status_code == 200:
            data = r.json()
            ucs = data.get("use_cases", [])
            if ucs:
                uc_id = ucs[0]["id"]
    response = admin_page.goto(f"{base_url}/studio/uc/{uc_id}")
    admin_page.wait_for_load_state("networkidle")
    content = admin_page.content()
    # Must not 500; 404 is acceptable if UC not configured at this path
    assert response.status != 500
    assert "Internal Server Error" not in content


def test_sync_runs_api_accessible(admin_page, app_server):
    """GET /studio/api/sync-runs returns 200 for authenticated admin."""
    base_url, _ = app_server
    tok = _get_tok(admin_page)
    with httpx.Client(base_url=base_url, cookies={"kvforge_session": tok}) as client:
        r = client.get("/studio/api/sync-runs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_sync_history_table_renders_on_connectors_page(admin_page, app_server):
    """Connectors page renders the history table DOM element."""
    base_url, _ = app_server
    tok = _get_tok(admin_page)
    # Trigger a sync so there's something to show
    with httpx.Client(base_url=base_url, cookies={"kvforge_session": tok}) as client:
        r = client.post("/studio/api/connectors", json={
            "type": "s3", "name": f"Monitor S3 {uuid.uuid4()}",
            "credentials": {"access_key_id": "A", "secret": "S", "bucket": "b"},
        })
        cid = r.json()["id"]
        client.post(f"/studio/api/connectors/{cid}/scopes",
                    json={"uc_id": "uc-mon", "scope_config": {"bucket": "b", "prefix": ""}})
        client.post(f"/studio/api/connectors/{cid}/sync")
        time.sleep(1.5)

    response = admin_page.goto(f"{base_url}/studio/connectors")
    admin_page.wait_for_load_state("networkidle")
    # Wait for JS to render history (it fetches on load)
    admin_page.wait_for_timeout(2000)
    assert response.status != 500
    assert "Internal Server Error" not in admin_page.content()
    # The history table element must exist in the DOM
    assert admin_page.locator("#history-tbody").count() == 1
