import os
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"


def test_portal_has_auth_login_route(tmp_path):
    import db.store as store
    store.DB_PATH = tmp_path / "test.db"
    store.close()
    import kvforge_portal
    from fastapi.testclient import TestClient
    client = TestClient(kvforge_portal.app, raise_server_exceptions=False)
    r = client.get("/auth/login")
    assert r.status_code == 200
    assert b"KVForge Studio" in r.content


def test_unauthenticated_studio_redirects_to_login(tmp_path):
    import db.store as store
    store.DB_PATH = tmp_path / "p.db"
    store.close()
    import kvforge_portal
    from fastapi.testclient import TestClient
    client = TestClient(kvforge_portal.app, follow_redirects=False, raise_server_exceptions=False)
    r = client.get("/studio/")
    assert r.status_code == 302
    assert "/auth/login" in r.headers.get("location", "")
