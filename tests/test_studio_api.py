import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_wizard_validate_accepts_valid():
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(api_router)
    client = TestClient(app)
    resp = client.post("/api/wizard-validate", json={"step": "train", "epochs": 3, "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_wizard_validate_rejects_bad_epochs():
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(api_router)
    client = TestClient(app)
    resp = client.post("/api/wizard-validate", json={"step": "train", "epochs": 0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert any("epochs" in e.lower() for e in data["errors"])


def test_wizard_validate_rejects_unknown_step():
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(api_router)
    client = TestClient(app)
    resp = client.post("/api/wizard-validate", json={"step": "invalid_step"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False


def test_hub_html_has_wizard_overlay():
    hub_path = Path(__file__).parent.parent / "templates" / "studio" / "hub.html"
    html = hub_path.read_text()
    assert "wizard-overlay" in html


def test_hub_html_has_wizard_steps():
    hub_path = Path(__file__).parent.parent / "templates" / "studio" / "hub.html"
    html = hub_path.read_text()
    assert "wz-step1" in html
    assert "wz-step2" in html
    assert "wz-step3" in html
