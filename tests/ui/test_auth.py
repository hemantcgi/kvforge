# tests/ui/test_auth.py
"""Playwright tests for auth flows — login, invite, logout, role enforcement."""
import uuid
import httpx
import pytest


def test_login_shows_form(app_server, _browser):
    base_url, _ = app_server
    ctx = _browser.new_context()
    page = ctx.new_page()
    page.goto(f"{base_url}/auth/login")
    assert page.locator("input[name='email']").count() == 1
    assert page.locator("input[name='password']").count() == 1
    assert page.locator("button[type='submit']").count() >= 1
    ctx.close()


def test_login_page_title(app_server, _browser):
    base_url, _ = app_server
    ctx = _browser.new_context()
    page = ctx.new_page()
    page.goto(f"{base_url}/auth/login")
    content = page.content()
    assert "KVForge" in content or "Login" in content or "Sign" in content
    ctx.close()


def test_login_invalid_password(app_server, _browser):
    base_url, _ = app_server
    ctx = _browser.new_context()
    page = ctx.new_page()
    page.goto(f"{base_url}/auth/login")
    page.fill("input[name='email']", "nobody@nowhere.com")
    page.fill("input[name='password']", "wrongpassword")
    page.click("button[type='submit']")
    page.wait_for_load_state("networkidle")
    assert "/auth/login" in page.url or "Invalid" in page.content() or "invalid" in page.content()
    ctx.close()


def test_unauthenticated_studio_redirects(app_server, _browser):
    base_url, _ = app_server
    ctx = _browser.new_context()
    page = ctx.new_page()
    page.goto(f"{base_url}/studio/", wait_until="commit")
    page.wait_for_load_state("networkidle")
    assert "/auth/login" in page.url
    ctx.close()


def test_logout_clears_session(app_server, db_path, _browser):
    from tests.ui.conftest import _seed_user
    base_url, secret = app_server
    tok = _seed_user(db_path, f"logout-{uuid.uuid4()}@ui.com", "admin", secret)
    ctx = _browser.new_context()
    page = ctx.new_page()
    page.context.add_cookies([{
        "name": "kvforge_session", "value": tok,
        "domain": "127.0.0.1", "path": "/",
    }])
    page.goto(f"{base_url}/auth/logout")
    page.wait_for_load_state("networkidle")
    assert "/auth/login" in page.url
    ctx.close()


def test_viewer_cannot_access_connectors_page(viewer_page, app_server):
    base_url, _ = app_server
    viewer_page.goto(f"{base_url}/studio/connectors")
    viewer_page.wait_for_load_state("networkidle")
    assert "403" in viewer_page.content() or "/auth/login" in viewer_page.url


def test_admin_can_access_connectors_page(admin_page, app_server):
    base_url, _ = app_server
    admin_page.goto(f"{base_url}/studio/connectors")
    admin_page.wait_for_load_state("networkidle")
    assert "Connectors" in admin_page.content()


def test_unauthenticated_api_returns_401(app_server):
    base_url, _ = app_server
    r = httpx.get(f"{base_url}/studio/api/connectors")
    assert r.status_code == 401


def test_admin_user_management_page(admin_page, app_server):
    base_url, _ = app_server
    admin_page.goto(f"{base_url}/studio/admin/users")
    admin_page.wait_for_load_state("networkidle")
    assert admin_page.locator("body").count() == 1
    assert "500" not in admin_page.content()
