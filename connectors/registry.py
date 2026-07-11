# connectors/registry.py
import json, os, uuid
from base64 import urlsafe_b64encode
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
import db.store as store

_MASK = "●●●●●●"
_UNSET = object()
_SALT = b"kvforge-connector-creds-v1"


def _fernet() -> Fernet:
    raw = os.environ.get("KVFORGE_SECRET_KEY", "dev-secret-change-me")
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_SALT,
        info=b"fernet",
    ).derive(raw.encode())
    return Fernet(urlsafe_b64encode(key))


def _encrypt(data: dict) -> str:
    return _fernet().encrypt(json.dumps(data).encode()).decode()


def _decrypt(token: str) -> dict:
    return json.loads(_fernet().decrypt(token.encode()))


class ConnectorRegistry:

    def create(self, connector_type: str, name: str,
               credentials: dict, created_by: str,
               schedule_cron: str | None = None,
               webhook_secret: str | None = None) -> dict:
        cid = str(uuid.uuid4())
        enc = _encrypt(credentials)
        store.execute(
            "INSERT INTO connector_configs(id,type,name,credentials_json,"
            "schedule_cron,webhook_secret,created_by) VALUES(?,?,?,?,?,?,?)",
            (cid, connector_type, name, enc, schedule_cron, webhook_secret, created_by)
        )
        store.commit()
        return self._safe_row(store.fetchone("SELECT * FROM connector_configs WHERE id=?", (cid,)))

    def list_all(self) -> list[dict]:
        rows = store.fetchall("SELECT * FROM connector_configs ORDER BY created_at DESC")
        return [self._safe_row(r) for r in rows]

    def get(self, cid: str) -> dict | None:
        row = store.fetchone("SELECT * FROM connector_configs WHERE id=?", (cid,))
        return self._safe_row(row) if row else None

    def get_credentials(self, cid: str) -> dict:
        row = store.fetchone("SELECT credentials_json FROM connector_configs WHERE id=?", (cid,))
        if not row:
            raise KeyError(f"connector {cid} not found")
        try:
            return _decrypt(row["credentials_json"])
        except Exception as exc:
            raise ValueError(f"failed to decrypt credentials for connector {cid}") from exc

    def update(self, cid: str,
               credentials: dict | None = None,
               schedule_cron=_UNSET,
               webhook_secret=_UNSET,
               name: str | None = None) -> dict:
        row = store.fetchone("SELECT * FROM connector_configs WHERE id=?", (cid,))
        if not row:
            raise KeyError(f"connector {cid} not found")
        new_enc = _encrypt(credentials) if credentials is not None else row["credentials_json"]
        new_cron = row["schedule_cron"] if schedule_cron is _UNSET else schedule_cron
        new_ws = row["webhook_secret"] if webhook_secret is _UNSET else webhook_secret
        new_name = name or row["name"]
        store.execute(
            "UPDATE connector_configs SET name=?,credentials_json=?,schedule_cron=?,webhook_secret=? WHERE id=?",
            (new_name, new_enc, new_cron, new_ws, cid)
        )
        store.commit()
        return self._safe_row(store.fetchone("SELECT * FROM connector_configs WHERE id=?", (cid,)))

    def delete(self, cid: str) -> None:
        store.execute("DELETE FROM connector_configs WHERE id=?", (cid,))
        store.commit()

    def upsert_scope(self, connector_id: str, uc_id: str, scope_config: dict) -> None:
        scope_json = json.dumps(scope_config)
        store.execute(
            "INSERT OR REPLACE INTO connector_uc_scopes(connector_id,uc_id,scope_config_json) VALUES(?,?,?)",
            (connector_id, uc_id, scope_json)
        )
        store.commit()

    def list_scopes(self, connector_id: str) -> list[dict]:
        rows = store.fetchall("SELECT * FROM connector_uc_scopes WHERE connector_id=?", (connector_id,))
        return [{"connector_id": r["connector_id"], "uc_id": r["uc_id"],
                 "scope_config": json.loads(r["scope_config_json"]),
                 "last_sync_at": r["last_sync_at"],
                 "last_delta_token": r["last_delta_token"]} for r in rows]

    def delete_scope(self, connector_id: str, uc_id: str) -> None:
        store.execute("DELETE FROM connector_uc_scopes WHERE connector_id=? AND uc_id=?",
                      (connector_id, uc_id))
        store.commit()

    @staticmethod
    def _safe_row(row) -> dict:
        d = dict(row)
        d["credentials"] = _MASK
        d.pop("credentials_json", None)
        return d
