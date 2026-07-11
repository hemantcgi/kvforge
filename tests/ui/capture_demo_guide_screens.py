#!/usr/bin/env python3
"""
Capture screenshots for the 8 demo use-case guides.
Run once against a live KVForge Studio (port 8080).

Usage:
    python tests/ui/capture_demo_guide_screens.py
"""
import base64
import os
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
import jwt as pyjwt
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent.parent
SCREENSHOTS = ROOT / "tests" / "screenshots"
SCREENSHOTS.mkdir(exist_ok=True)

SECRET = "demo-guide-capture-secret-xxxxx"
DB_PATH = Path(os.environ.get("KVFORGE_DB_PATH", ROOT / "kvforge_studio.db"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def seed_user(db_path: Path, email: str, role: str, secret: str) -> str:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        existing = con.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
        if existing:
            uid = existing["id"]
        else:
            uid = str(uuid.uuid4())
            hashed = bcrypt.hashpw(b"demo123", bcrypt.gensalt(rounds=4)).decode()
            con.execute(
                "INSERT INTO users(id,email,hashed_pw,role,provider) VALUES(?,?,?,?,?)",
                (uid, email, hashed, role, "local"),
            )
        exp = datetime.now(timezone.utc) + timedelta(hours=2)
        tok = pyjwt.encode({"sub": uid, "role": role, "exp": exp}, secret, algorithm="HS256")
        sid = str(uuid.uuid4())
        con.execute(
            "INSERT OR IGNORE INTO sessions(id,user_id,jwt_token,expires_at) VALUES(?,?,?,?)",
            (sid, uid, tok, exp.isoformat()),
        )
        con.commit()
        return tok
    finally:
        con.close()


def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def shoot(page, name: str, desc: str) -> Path:
    path = SCREENSHOTS / f"demo_guide__{name}.png"
    page.screenshot(path=str(path), full_page=True)
    print(f"  ✓ {name}")
    return path


def main():
    port = free_port()
    db_path = ROOT / f"_guide_capture_{port}.db"
    secret = SECRET

    env = {**os.environ, "KVFORGE_SECRET_KEY": secret,
           "KVFORGE_DB_PATH": str(db_path), "KVFORGE_UI_TEST": "1"}
    proc = subprocess.Popen(
        [sys.executable, "kvforge_portal.py", "--port", str(port)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"

    import httpx
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            httpx.get(f"{base}/auth/login", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("Server did not start")

    admin_tok = seed_user(db_path, "admin@demo.com", "admin", secret)
    editor_tok = seed_user(db_path, "editor@demo.com", "editor", secret)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()

        print("Capturing login flow…")
        page.goto(f"{base}/studio/")
        page.wait_for_load_state("networkidle")
        shoot(page, "01_login_page", "Login page — unauthenticated redirect")

        page.fill("input[name='email']", "admin@demo.com")
        page.fill("input[name='password']", "demo123")
        shoot(page, "02_login_filled", "Login form with credentials filled in")

        print("Capturing Studio Hub (empty state)…")
        ctx.add_cookies([{"name": "kvforge_session", "value": admin_tok,
                          "domain": "127.0.0.1", "path": "/"}])
        page.goto(f"{base}/studio/")
        page.wait_for_load_state("networkidle")
        shoot(page, "03_hub_empty", "Studio Hub — empty state, no use cases yet")

        print("Capturing GPU connect page…")
        page.goto(f"{base}/studio/gpu-connect")
        page.wait_for_load_state("networkidle")
        shoot(page, "04_gpu_connect", "Remote GPU Connection wizard — Step 1")

        print("Capturing GPU connect Step 1 filled…")
        page.fill("#inp-host", "54.198.243.26")
        page.fill("#inp-user", "ubuntu")
        page.fill("#inp-name", "A10G Production")
        shoot(page, "05_gpu_connect_filled", "GPU connect form with host details filled in")

        print("Capturing Addons (connectors) page…")
        page.goto(f"{base}/studio/connectors")
        page.wait_for_load_state("networkidle")
        shoot(page, "06_addons_page", "Addons page — manage data source connectors")

        print("Capturing Settings page…")
        page.goto(f"{base}/studio/settings")
        page.wait_for_load_state("networkidle")
        shoot(page, "07_settings_page", "Settings page — LLM and environment configuration")

        print("Capturing Wizard…")
        page.goto(f"{base}/studio/wizard")
        page.wait_for_load_state("networkidle")
        shoot(page, "08_wizard_step1", "New Use Case Wizard — Step 1: Data Source")

        # Try clicking HuggingFace source if available
        try:
            page.click("text=HuggingFace", timeout=2000)
            page.wait_for_timeout(500)
            shoot(page, "09_wizard_hf_selected", "Wizard Step 1 — HuggingFace source type selected")
        except Exception:
            pass

        # Try clicking PDF source
        try:
            page.goto(f"{base}/studio/wizard")
            page.wait_for_load_state("networkidle")
            page.click("text=PDF", timeout=2000)
            page.wait_for_timeout(500)
            shoot(page, "10_wizard_pdf_selected", "Wizard Step 1 — PDF / File Upload source type selected")
        except Exception:
            pass

        # Hub with GPU warning shown
        print("Capturing hub with GPU banner…")
        page.goto(f"{base}/studio/")
        page.wait_for_load_state("networkidle")
        # GPU warning is already shown when no nvidia-smi
        shoot(page, "11_hub_gpu_warning", "Studio Hub showing 'No GPU Detected' warning banner")

        browser.close()

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    # Clean up temp db
    try:
        db_path.unlink()
    except Exception:
        pass

    print(f"\nDone. Screenshots in: {SCREENSHOTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
