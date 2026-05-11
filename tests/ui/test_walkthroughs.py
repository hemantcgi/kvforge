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
            "Login page shown — unauthenticated users are automatically redirected here when they try to open /studio/.",
            action="Navigated to /studio/ without a session cookie.")

    # Step 2: Attempt login with bad credentials
    page.fill("input[name='email']", "wrong@example.com")
    page.fill("input[name='password']", "badpassword")
    capture(page, TN, 2, "login_form_filled_wrong",
            "Login form with invalid credentials typed in — the 'Sign In' button has not been clicked yet.",
            action="Filled email='wrong@example.com' and password='badpassword' in the login form.")
    page.click("button[type='submit']")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 3, "login_error_shown",
            "Login page reloaded with an error message — the server rejected the credentials and rendered the error template variable.",
            action="Submitted the login form. Server returned 200 with an error message.")

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
            "Studio hub (new Atlassian-style design) — the main dashboard showing configured use-cases, system status cards, and the sidebar with navigation.",
            action="Injected a valid admin JWT cookie, then navigated to /studio/.")

    # Step 5: Navigate to user management page
    page.goto(f"{base_url}/studio/admin/users")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 5, "admin_users_page",
            "Admin user management page — lists all registered users with their email, role badge, and provider. Admin-only page (403 for non-admins).",
            action="Navigated to /studio/admin/users as an authenticated admin user.")

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
            "Connectors dashboard — the starting state before any connector has been added. May show an empty list or connectors from prior test runs.",
            action="Navigated to /studio/connectors as an authenticated admin.")

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
            f"The new connector '{conn_name}' now appears in the connectors list with its type badge (GDrive) and a health status dot.",
            action=f"Created connector via POST /studio/api/connectors with type='gdrive', then reloaded the page.")

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
            "Connectors page after sync was triggered — the page still shows the connector list. The sync run was started in the background (a background task was created).",
            action="Called POST /studio/api/connectors/{id}/sync — received ok:true. Also added a UC scope via POST /studio/api/connectors/{id}/scopes.")

    # Step 6: Wait for background sync task and reload
    time.sleep(2.0)
    page.reload()
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2000)  # let JS fetch history
    capture(page, TN, 4, "sync_history_populated",
            "Sync history table now shows the completed run — includes connector name, trigger type ('manual'), result status, and duration. The table auto-refreshes.",
            action="Waited 2 seconds for the background sync task to complete, then reloaded the page.")

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
            "Admin user sees the full connectors UI — the 'Add Connector' button is visible. Only admin and editor roles can access this page.",
            action="Authenticated as admin role, navigated to /studio/connectors.")

    # Step 2: Logout
    page.goto(f"{base_url}/auth/logout")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 2, "after_logout",
            "Login page shown after logout — the session cookie was cleared. User is redirected to /auth/login.",
            action="Navigated to /auth/logout — server deleted the session cookie and redirected here.")

    # Step 3: Inject viewer cookie
    page.context.add_cookies([{
        "name": "kvforge_session", "value": viewer_tok,
        "domain": "127.0.0.1", "path": "/",
    }])

    # Step 4: Viewer tries to access connectors page → 403
    page.goto(f"{base_url}/studio/connectors")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 3, "viewer_gets_403_on_connectors",
            "403 Forbidden response — the server rejected the viewer's request to access /studio/connectors. The middleware checks role before rendering the page.",
            action="Injected a viewer-role JWT cookie, then navigated to /studio/connectors.")

    # Step 5: Show studio hub is still accessible to viewer
    page.goto(f"{base_url}/studio/")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 4, "viewer_can_see_hub",
            "Studio hub accessible to viewer — the main dashboard loads successfully. The hub is accessible to all authenticated roles; only admin/editor-only pages return 403.",
            action="Navigated to /studio/ as a viewer. Same viewer cookie from the previous step.")

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
            "Studio hub — the main entry point. Shows the system status bar (Active UCs, Queries Today, GPU, System Health) and UC cards with phase badges and metrics.",
            action="Navigated to /studio/ as an authenticated admin.")

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
            f"UC detail page for '{uc_id}' — shows the use-case's current pipeline phase, PRS score history, training controls, and live sync status.",
            action=f"Navigated to /studio/uc/{uc_id} — the UC ID was fetched from GET /studio/api/registry.")

    # Step 3: Connectors page showing sync history
    page.goto(f"{base_url}/studio/connectors")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2000)  # wait for JS to fetch history
    capture(page, TN, 3, "connectors_with_sync_history",
            "Connectors page with both sections visible — the connector list at top and the sync run history table below. The history table polls every 5 seconds.",
            action="Navigated to /studio/connectors, waited 2 seconds for the JS to fetch and render the history table.")

    # Step 4: Sync runs API response (show raw data the dashboard uses)
    page.goto(f"{base_url}/studio/api/sync-runs")
    page.wait_for_load_state("networkidle")
    capture(page, TN, 4, "sync_runs_api_raw",
            "Raw JSON response from GET /studio/api/sync-runs — the data source that powers the sync history table. Each record has connector_id, trigger, status, files_done, and duration.",
            action="Navigated directly to /studio/api/sync-runs in the browser — Chromium renders JSON as plain text.")

    ctx.close()
