# tests/ui/conftest.py
"""Shared fixtures for Playwright UI tests."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import jwt as pyjwt
import pytest
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent.parent
SCREENSHOTS = ROOT / "tests" / "screenshots"
SCREENSHOTS.mkdir(exist_ok=True)


# ── Server lifecycle ──────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("ui_db") / "studio.db"


@pytest.fixture(scope="session")
def app_server(db_path):
    port = _free_port()
    secret = "ui-test-secret-32bytesXXXXXXXXX"
    env = {
        **os.environ,
        "KVFORGE_SECRET_KEY": secret,
        "KVFORGE_DB_PATH": str(db_path),
        "KVFORGE_UI_TEST": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "kvforge_portal.py", "--port", str(port)],
        cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}/auth/login", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        out, err = proc.stdout.read(), proc.stderr.read()
        proc.kill()
        raise RuntimeError(f"Server did not start in time.\nSTDOUT: {out.decode()}\nSTDERR: {err.decode()}")

    base_url = f"http://127.0.0.1:{port}"
    yield base_url, secret
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── JWT cookie helpers ────────────────────────────────────────────────────────

def _seed_user(db_path: Path, email: str, role: str, secret: str) -> str:
    """Insert a user directly into the test DB; return a signed JWT.

    If a user with this email already exists, reuse their id so that the JWT
    sub claim always matches the users table (avoids 401 on repeated calls
    with the same email across function-scoped fixtures).
    """
    import sqlite3, bcrypt
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    existing = con.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if existing:
        uid = existing["id"]
    else:
        uid = str(uuid.uuid4())
        hashed = bcrypt.hashpw(b"testpass123", bcrypt.gensalt(rounds=4)).decode()
        con.execute(
            "INSERT INTO users(id,email,hashed_pw,role,provider) VALUES(?,?,?,?,?)",
            (uid, email, hashed, role, "local")
        )
    exp = datetime.now(timezone.utc) + timedelta(hours=2)
    tok = pyjwt.encode({"sub": uid, "role": role, "exp": exp}, secret, algorithm="HS256")
    sid = str(uuid.uuid4())
    con.execute(
        "INSERT OR IGNORE INTO sessions(id,user_id,jwt_token,expires_at) VALUES(?,?,?,?)",
        (sid, uid, tok, exp.isoformat())
    )
    con.commit()
    con.close()
    return tok


def _authed_page(page: Page, base_url: str, tok: str) -> Page:
    page.goto(f"{base_url}/auth/login")
    page.context.add_cookies([{
        "name": "kvforge_session", "value": tok,
        "domain": "127.0.0.1", "path": "/",
    }])
    return page


# ── Role-specific page fixtures ───────────────────────────────────────────────

@pytest.fixture(scope="session")
def _playwright_session():
    with sync_playwright() as pw:
        yield pw


@pytest.fixture(scope="session")
def _browser(_playwright_session):
    browser = _playwright_session.chromium.launch(headless=True)
    yield browser
    browser.close()


@pytest.fixture
def admin_page(app_server, db_path, _browser):
    base_url, secret = app_server
    tok = _seed_user(db_path, "admin@ui-test.com", "admin", secret)
    ctx = _browser.new_context()
    page = ctx.new_page()
    _authed_page(page, base_url, tok)
    page._base_url = base_url
    yield page
    ctx.close()


@pytest.fixture
def editor_page(app_server, db_path, _browser):
    base_url, secret = app_server
    tok = _seed_user(db_path, "editor@ui-test.com", "editor", secret)
    ctx = _browser.new_context()
    page = ctx.new_page()
    _authed_page(page, base_url, tok)
    page._base_url = base_url
    yield page
    ctx.close()


@pytest.fixture
def viewer_page(app_server, db_path, _browser):
    base_url, secret = app_server
    tok = _seed_user(db_path, "viewer@ui-test.com", "viewer", secret)
    ctx = _browser.new_context()
    page = ctx.new_page()
    _authed_page(page, base_url, tok)
    page._base_url = base_url
    yield page
    ctx.close()


# ── Connector mocks ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_gdrive(monkeypatch):
    """Patch GDriveConnector so no real API calls are made."""
    from connectors.base import SourceFile
    from datetime import datetime as _dt

    fake_files = [
        SourceFile("f1", "doc1.pdf", "shared/doc1.pdf", 1024,
                   _dt(2026, 5, 1), "application/pdf"),
        SourceFile("f2", "faq.md", "shared/faq.md", 512,
                   _dt(2026, 5, 2), "text/plain"),
        SourceFile("f3", "report.docx", "shared/report.docx", 2048,
                   _dt(2026, 5, 3),
                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ]

    class FakeGDrive:
        def list_files(self): return fake_files
        def download(self, f): return f"Content of {f.name}".encode()
        def get_modified_at(self, f): return f.modified_at
        def supports_delta(self): return False
        def get_delta(self, token): return fake_files, "delta-token-001"

    try:
        import connectors.gdrive_connector as gd
        monkeypatch.setattr(gd, "GDriveConnector", lambda creds: FakeGDrive())
    except ImportError:
        pass
    return FakeGDrive


@pytest.fixture
def mock_s3(monkeypatch):
    from connectors.base import SourceFile
    from datetime import datetime as _dt

    fake_files = [
        SourceFile("s1", "guide.pdf", "docs/guide.pdf", 4096, _dt(2026, 4, 10), "application/pdf"),
    ]

    class FakeS3:
        def list_files(self): return fake_files
        def download(self, f): return b"AWS doc content"
        def get_modified_at(self, f): return f.modified_at
        def supports_delta(self): return False
        def get_delta(self, token): return fake_files, None

    try:
        import connectors.s3_connector as s3m
        monkeypatch.setattr(s3m, "S3Connector", lambda creds: FakeS3())
    except ImportError:
        pass
    return FakeS3


@pytest.fixture
def mock_sharepoint(monkeypatch):
    from connectors.base import SourceFile
    from datetime import datetime as _dt

    fake_files = [
        SourceFile("sp1", "policy.docx", "/sites/hr/policy.docx", 2000,
                   _dt(2026, 3, 15),
                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        SourceFile("sp2", "handbook.pdf", "/sites/hr/handbook.pdf", 8000,
                   _dt(2026, 3, 20), "application/pdf"),
    ]

    class FakeSP:
        def list_files(self): return fake_files
        def download(self, f): return b"SharePoint doc content"
        def get_modified_at(self, f): return f.modified_at
        def supports_delta(self): return True
        def get_delta(self, token): return fake_files, "sp-delta-token-99"

    try:
        import connectors.sharepoint_connector as spm
        monkeypatch.setattr(spm, "SharePointConnector", lambda creds: FakeSP())
    except ImportError:
        pass
    return FakeSP


# ── Screenshot on failure ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def screenshot_on_failure(request):
    yield
    rep = getattr(request.node, "rep_call", None)
    if rep is not None and rep.failed:
        page = None
        for fixture_name in request.fixturenames:
            try:
                val = request.getfixturevalue(fixture_name)
                if hasattr(val, "screenshot"):
                    page = val
                    break
            except Exception:
                pass
        if page:
            safe = request.node.name.replace("/", "_").replace(" ", "_")
            path = SCREENSHOTS / f"{safe}.png"
            try:
                page.screenshot(path=str(path), full_page=True)
                print(f"\nScreenshot saved: {path}")
            except Exception:
                pass


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


# ── Walkthrough screenshot helpers ────────────────────────────────────────────

import json as _json

_MANIFEST_PATH = ROOT / "tests" / "walkthrough_manifest.json"
_manifest: list[dict] = []


def capture(page: Page, test_name: str, step_index: int, step_slug: str, description: str) -> str:
    """Take a screenshot and record metadata to the walkthrough manifest."""
    filename = f"{test_name}__{step_index:02d}_{step_slug}.png"
    path = SCREENSHOTS / filename
    page.screenshot(path=str(path), full_page=True)
    _manifest.append({
        "test": test_name,
        "step": step_index,
        "slug": step_slug,
        "description": description,
        "file": filename,
    })
    return str(path)


@pytest.fixture(scope="session", autouse=True)
def _write_manifest():
    _manifest.clear()
    yield
    _MANIFEST_PATH.write_text(_json.dumps(_manifest, indent=2))
