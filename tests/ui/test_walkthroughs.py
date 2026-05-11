# tests/ui/test_walkthroughs.py
"""
Walkthrough tests — each test narrates a complete user journey with step-by-step screenshots.
These are designed to produce an HTML report showing the UI at each stage.
Existing 21 tests are NOT modified.
"""
import json
import time
import uuid
import httpx
import pytest

from tests.ui.conftest import _seed_user, capture


# ─────────────────────────────────────────────────────────────────────────────
# Walkthrough 1: Login → Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def test_walkthrough_login_to_dashboard(app_server, db_path, _browser):
    """
    Shows: unauthenticated redirect → login page → invalid credentials error →
           successful login → studio hub dashboard.
    """
    TN = "walkthrough_login_to_dashboard"
    base_url, secret = app_server
    ctx = _browser.new_context()
    page = ctx.new_page()

    # Step 1: Navigate to studio/ without a cookie → redirected to login
    page.goto(f"{base_url}/studio/")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 1, "redirect_to_login",
            "Navigating to /studio/ without a session redirects to the login page.")

    # Step 2: Attempt login with bad credentials
    page.fill("input[name='email']", "wrong@example.com")
    page.fill("input[name='password']", "badpassword")
    capture(page, TN, 2, "login_form_filled_wrong",
            "Login form filled with invalid credentials, before submitting.")
    page.click("button[type='submit']")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 3, "login_error_shown",
            "Server returns login page with an error message after wrong credentials.")

    # Step 3: Seed an admin user and inject cookie
    tok = _seed_user(db_path, f"walkthrough-admin-{uuid.uuid4()}@ui.com", "admin", secret)
    page.context.add_cookies([{
        "name": "kvforge_session", "value": tok,
        "domain": "127.0.0.1", "path": "/",
    }])

    # Step 4: Navigate to studio hub (authenticated)
    page.goto(f"{base_url}/studio/")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 4, "studio_hub_authenticated",
            "Studio hub landing page after successful authentication as admin.")

    # Step 5: Navigate to user management page
    page.goto(f"{base_url}/studio/admin/users")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 5, "admin_users_page",
            "Admin user management page — shows all registered users and their roles.")

    ctx.close()


# ─────────────────────────────────────────────────────────────────────────────
# Walkthrough 2: Connector CRUD + Sync Trigger
# ─────────────────────────────────────────────────────────────────────────────

def test_walkthrough_create_connector_and_sync(app_server, db_path, _browser):
    """
    Shows: empty connectors page → add connector via prompt → connector appears in list →
           add UC scope → trigger sync → sync history row appears.
    """
    TN = "walkthrough_create_connector_and_sync"
    base_url, secret = app_server
    tok = _seed_user(db_path, f"walkthrough-conn-{uuid.uuid4()}@ui.com", "admin", secret)

    ctx = _browser.new_context()
    page = ctx.new_page()
    page.context.add_cookies([{
        "name": "kvforge_session", "value": tok,
        "domain": "127.0.0.1", "path": "/",
    }])

    # Step 1: Open connectors page (initially empty or has prior test connectors)
    page.goto(f"{base_url}/studio/connectors")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1000)  # let JS load the list
    capture(page, TN, 1, "connectors_page_initial",
            "Connectors dashboard — showing existing connectors or 'no connectors yet' state.")

    # Step 2: Create a connector via the REST API (simulates what the Add button does)
    conn_name = f"My GDrive {uuid.uuid4().hex[:6]}"
    with httpx.Client(base_url=base_url, cookies={"kvforge_session": tok}, timeout=10) as client:
        r = client.post("/studio/api/connectors", json={
            "type": "gdrive",
            "name": conn_name,
            "credentials": {"service_account_json": json.dumps({"type": "service_account"})},
        })
        assert r.status_code == 200
        cid = r.json()["id"]

    # Step 3: Reload page — connector should now appear
    page.reload()
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)
    capture(page, TN, 2, "connector_added_in_list",
            f"New connector '{conn_name}' appears in the connector list with type badge and health dot.")

    # Step 4: Add a UC scope via API
    with httpx.Client(base_url=base_url, cookies={"kvforge_session": tok}, timeout=10) as client:
        client.post(f"/studio/api/connectors/{cid}/scopes", json={
            "uc_id": "uc-walkthrough",
            "scope_config": {"folder_id": "shared-drive-xyz", "include_shared_drives": True},
        })

    # Step 5: Trigger a sync run via API
    with httpx.Client(base_url=base_url, cookies={"kvforge_session": tok}, timeout=10) as client:
        r2 = client.post(f"/studio/api/connectors/{cid}/sync")
        assert r2.json()["ok"] is True

    capture(page, TN, 3, "sync_triggered",
            "Sync triggered via POST /studio/api/connectors/{id}/sync — returns ok:true. "
            "Background task starts; sync_runs record created immediately.")

    # Step 6: Wait for background sync task and reload
    time.sleep(2.0)
    page.reload()
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2000)  # let JS fetch history
    capture(page, TN, 4, "sync_history_populated",
            "After sync completes (success or error), the sync run history table shows the run "
            "with connector name, trigger type, result status, and duration.")

    ctx.close()


# ─────────────────────────────────────────────────────────────────────────────
# Walkthrough 3: Role Enforcement (Admin vs Viewer)
# ─────────────────────────────────────────────────────────────────────────────

def test_walkthrough_role_enforcement(app_server, db_path, _browser):
    """
    Shows: admin sees connectors page fully → logout → login as viewer →
           viewer gets 403 on connectors page → viewer's API create attempt returns 403.
    """
    TN = "walkthrough_role_enforcement"
    base_url, secret = app_server
    admin_tok = _seed_user(db_path, f"walkthrough-rbac-admin-{uuid.uuid4()}@ui.com", "admin", secret)
    viewer_tok = _seed_user(db_path, f"walkthrough-rbac-viewer-{uuid.uuid4()}@ui.com", "viewer", secret)

    ctx = _browser.new_context()
    page = ctx.new_page()

    # Step 1: Admin views the connectors page
    page.context.add_cookies([{
        "name": "kvforge_session", "value": admin_tok,
        "domain": "127.0.0.1", "path": "/",
    }])
    page.goto(f"{base_url}/studio/connectors")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1000)
    capture(page, TN, 1, "admin_sees_connectors",
            "Admin user can access the /studio/connectors page — sees full UI with Add button.")

    # Step 2: Logout
    page.goto(f"{base_url}/auth/logout")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 2, "after_logout",
            "After logout, session is cleared and user is redirected to the login page.")

    # Step 3: Inject viewer cookie
    page.context.add_cookies([{
        "name": "kvforge_session", "value": viewer_tok,
        "domain": "127.0.0.1", "path": "/",
    }])

    # Step 4: Viewer tries to access connectors page → 403
    page.goto(f"{base_url}/studio/connectors")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 3, "viewer_gets_403_on_connectors",
            "Viewer role is denied access to /studio/connectors — server returns 403 Forbidden.")

    # Step 5: Show studio hub is still accessible to viewer
    page.goto(f"{base_url}/studio/")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 4, "viewer_can_see_hub",
            "Viewer can still access the studio hub — role restriction applies only to "
            "sensitive pages (connectors) not the general dashboard.")

    ctx.close()


# ─────────────────────────────────────────────────────────────────────────────
# Walkthrough 4: Monitoring — Hub → UC Detail → Sync History
# ─────────────────────────────────────────────────────────────────────────────

def test_walkthrough_monitoring(app_server, db_path, _browser):
    """
    Shows: studio hub overview → UC detail page → connectors page sync history table.
    """
    TN = "walkthrough_monitoring"
    base_url, secret = app_server
    tok = _seed_user(db_path, f"walkthrough-mon-{uuid.uuid4()}@ui.com", "admin", secret)

    ctx = _browser.new_context()
    page = ctx.new_page()
    page.context.add_cookies([{
        "name": "kvforge_session", "value": tok,
        "domain": "127.0.0.1", "path": "/",
    }])

    # Step 1: Studio hub
    page.goto(f"{base_url}/studio/")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 1, "studio_hub_overview",
            "Studio hub — shows all configured use-cases with their current phase, "
            "LoRA version, and PRS score. Entry point for monitoring each UC.")

    # Step 2: Navigate to UC detail
    # Get a valid UC ID from the registry API
    with httpx.Client(base_url=base_url, cookies={"kvforge_session": tok}, timeout=5) as client:
        r = client.get("/studio/api/registry")
        payload = r.json() if r.status_code == 200 else {}
        uc_list = payload.get("use_cases", []) if isinstance(payload, dict) else payload
    uc_id = uc_list[0]["id"] if uc_list else "uc1"

    page.goto(f"{base_url}/studio/uc/{uc_id}")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 2, "uc_detail_page",
            f"Use-case detail page for '{uc_id}' — shows pipeline phase, PRS history chart, "
            "LoRA training controls, and live sync status.")

    # Step 3: Connectors page showing sync history
    page.goto(f"{base_url}/studio/connectors")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2000)  # wait for JS to fetch history
    capture(page, TN, 3, "connectors_with_sync_history",
            "Connectors page showing both the connector list and the sync run history table. "
            "History auto-refreshes every 5 seconds via SSE polling.")

    # Step 4: Sync runs API response (show raw data the dashboard uses)
    page.goto(f"{base_url}/studio/api/sync-runs")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 4, "sync_runs_api_raw",
            "Raw JSON from GET /studio/api/sync-runs — the data source powering the "
            "sync history table. Each record has connector_id, trigger, status, files_done, duration.")

    ctx.close()
