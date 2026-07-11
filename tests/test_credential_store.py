import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_local_store_set_get(tmp_path):
    from connectors.credential_store import LocalFileCredentialStore
    store = LocalFileCredentialStore(path=str(tmp_path / "creds.json"))
    store.set("my_key", "my_secret")
    assert store.get("my_key") == "my_secret"


def test_local_store_missing_key_returns_none(tmp_path):
    from connectors.credential_store import LocalFileCredentialStore
    store = LocalFileCredentialStore(path=str(tmp_path / "creds.json"))
    assert store.get("nonexistent") is None


def test_local_store_delete(tmp_path):
    from connectors.credential_store import LocalFileCredentialStore
    store = LocalFileCredentialStore(path=str(tmp_path / "creds.json"))
    store.set("k", "v")
    store.delete("k")
    assert store.get("k") is None


def test_local_store_persists_across_instances(tmp_path):
    from connectors.credential_store import LocalFileCredentialStore
    path = str(tmp_path / "creds.json")
    LocalFileCredentialStore(path=path).set("token", "abc")
    assert LocalFileCredentialStore(path=path).get("token") == "abc"


def test_credential_store_protocol_satisfied(tmp_path):
    from connectors.credential_store import LocalFileCredentialStore, CredentialStore
    store = LocalFileCredentialStore(path=str(tmp_path / "creds.json"))
    assert isinstance(store, CredentialStore)


def test_delete_nonexistent_key_is_safe(tmp_path):
    from connectors.credential_store import LocalFileCredentialStore
    store = LocalFileCredentialStore(path=str(tmp_path / "creds.json"))
    store.delete("never_set")  # must not raise


def test_overwrite_existing_key(tmp_path):
    from connectors.credential_store import LocalFileCredentialStore
    store = LocalFileCredentialStore(path=str(tmp_path / "creds.json"))
    store.set("k", "v1")
    store.set("k", "v2")
    assert store.get("k") == "v2"
