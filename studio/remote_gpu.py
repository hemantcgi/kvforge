# studio/remote_gpu.py
"""Remote GPU connection: SSH via paramiko, encrypted profile storage, setup streaming."""

import json
import os
import re
import uuid
from base64 import urlsafe_b64encode
from io import StringIO
from pathlib import Path
from typing import Generator

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.fernet import Fernet

_SALT = b"kvforge-remote-gpu-v1"
_PROFILES_PATH = Path.home() / ".kvforge" / "gpu_profiles.json"

_SETUP_COMMANDS = [
    ("Updating package lists", "sudo apt-get update -q"),
    ("Installing Python build deps", "sudo apt-get install -y python3-pip python3-venv build-essential -q"),
    ("Installing PyTorch (CUDA 12.1)", "pip3 install torch --index-url https://download.pytorch.org/whl/cu121 -q"),
    ("Installing transformers + peft", "pip3 install transformers peft bitsandbytes accelerate datasets -q"),
    ("Installing FastAPI + uvicorn", "pip3 install fastapi uvicorn httpx -q"),
    ("Installing vector store clients", "pip3 install qdrant-client chromadb faiss-cpu fastembed -q"),
    ("Installing KVForge extras", "pip3 install anthropic apscheduler msal requests paramiko cryptography -q"),
    ("Verifying CUDA drivers", "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
]

_HOST_RE = re.compile(r'^[a-zA-Z0-9._\-]{1,253}$')
_USER_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,32}$')


# ── Encryption ────────────────────────────────────────────────────────────────

def _fernet() -> Fernet:
    raw = os.environ.get("KVFORGE_SECRET_KEY", "dev-secret-change-me")
    key = HKDF(
        algorithm=hashes.SHA256(), length=32,
        salt=_SALT, info=b"fernet",
    ).derive(raw.encode())
    return Fernet(urlsafe_b64encode(key))


def _encrypt(data: str) -> str:
    return _fernet().encrypt(data.encode()).decode()


def _decrypt(token: str) -> str:
    return _fernet().decrypt(token.encode()).decode()


# ── Profile CRUD ──────────────────────────────────────────────────────────────

def _load_profiles() -> list[dict]:
    if not _PROFILES_PATH.exists():
        return []
    try:
        return json.loads(_PROFILES_PATH.read_text())
    except Exception:
        return []


def _save_profiles(profiles: list[dict]) -> None:
    _PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROFILES_PATH.write_text(json.dumps(profiles, indent=2))


def list_profiles() -> list[dict]:
    """Return profiles with pem_key omitted."""
    return [
        {k: v for k, v in p.items() if k != "pem_enc"}
        for p in _load_profiles()
    ]


def save_profile(profile_id: str, host: str, user: str, port: int,
                 display_name: str, pem_key: str, fingerprint: str) -> dict:
    profiles = _load_profiles()
    entry = {
        "id": profile_id,
        "display_name": display_name,
        "host": host,
        "user": user,
        "port": port,
        "fingerprint": fingerprint,
        "pem_enc": _encrypt(pem_key),
    }
    profiles = [p for p in profiles if p["id"] != profile_id]
    profiles.append(entry)
    _save_profiles(profiles)
    return {k: v for k, v in entry.items() if k != "pem_enc"}


def delete_profile(profile_id: str) -> bool:
    profiles = _load_profiles()
    new = [p for p in profiles if p["id"] != profile_id]
    if len(new) == len(profiles):
        return False
    _save_profiles(new)
    return True


def _get_pem(profile_id: str) -> str | None:
    for p in _load_profiles():
        if p["id"] == profile_id:
            return _decrypt(p["pem_enc"])
    return None


# ── SSH helpers ───────────────────────────────────────────────────────────────

def _load_pkey(pem_content: str):
    """Load a PEM private key; works with paramiko 3.x and 4.x."""
    import paramiko
    # paramiko 4.x: PKey.from_private_key() auto-detects key type
    if hasattr(paramiko.PKey, 'from_private_key'):
        return paramiko.PKey.from_private_key(StringIO(pem_content))
    # paramiko 3.x fallback: try each key class in turn
    for cls in (paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey):
        try:
            return cls.from_private_key(StringIO(pem_content))
        except Exception:
            continue
    raise ValueError("Could not parse PEM key — unsupported key type")


def _make_client(host: str, user: str, port: int, pem_content: str):
    import paramiko
    pkey = _load_pkey(pem_content)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, port=port, pkey=pkey, timeout=15)
    return client


def _run_command(client, cmd: str) -> tuple[int, str, str]:
    """Run a single command; return (exit_code, stdout, stderr)."""
    _, stdout, stderr = client.exec_command(cmd, timeout=300)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


# ── In-memory sessions ────────────────────────────────────────────────────────

_sessions: dict[str, dict] = {}


def create_session(host: str, user: str, port: int, pem_key: str, display_name: str) -> str:
    if not _HOST_RE.match(host):
        raise ValueError(f"Invalid host: {host!r}")
    if not _USER_RE.match(user):
        raise ValueError(f"Invalid user: {user!r}")
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}")
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "id": session_id,
        "host": host,
        "user": user,
        "port": port,
        "pem_key": pem_key,
        "display_name": display_name or host,
        "fingerprint": None,
    }
    return session_id


def get_session(session_id: str) -> dict | None:
    return _sessions.get(session_id)


def drop_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


# ── Streaming generators (sync; wrap in thread for SSE) ──────────────────────

def stream_test_connection(session_id: str) -> Generator[dict, None, None]:
    """Connect and run nvidia-smi; yield SSE-style dicts."""
    sess = _sessions.get(session_id)
    if not sess:
        yield {"type": "error", "message": "Session not found"}
        return

    yield {"type": "step", "message": f"Connecting to {sess['user']}@{sess['host']}:{sess['port']} …"}
    try:
        client = _make_client(sess["host"], sess["user"], sess["port"], sess["pem_key"])
    except Exception as exc:
        yield {"type": "error", "message": f"SSH connection failed: {exc}"}
        return

    # Capture host fingerprint
    transport = client.get_transport()
    if transport:
        key = transport.get_remote_server_key()
        sess["fingerprint"] = key.get_fingerprint().hex(":")
        yield {"type": "info", "message": f"Host fingerprint: {sess['fingerprint']}"}

    yield {"type": "step", "message": "Running nvidia-smi …"}
    code, out, err = _run_command(client, "nvidia-smi")
    if code != 0:
        client.close()
        yield {"type": "error", "message": f"nvidia-smi failed (exit {code}):\n{err.strip() or 'command not found'}"}
        return

    yield {"type": "output", "message": out.strip()}
    yield {"type": "step", "message": "Checking Python environment …"}
    _, pyout, _ = _run_command(client, "python3 --version 2>&1 || python --version 2>&1")
    yield {"type": "info", "message": pyout.strip() or "python not found"}
    client.close()
    yield {"type": "done", "message": "Connection verified successfully"}


def stream_setup_gpu(session_id: str) -> Generator[dict, None, None]:
    """Run KVForge dependency installation; yield SSE-style dicts."""
    sess = _sessions.get(session_id)
    if not sess:
        yield {"type": "error", "message": "Session not found"}
        return

    yield {"type": "step", "message": f"Connecting to {sess['user']}@{sess['host']} …"}
    try:
        client = _make_client(sess["host"], sess["user"], sess["port"], sess["pem_key"])
    except Exception as exc:
        yield {"type": "error", "message": f"SSH connection failed: {exc}"}
        return

    for label, cmd in _SETUP_COMMANDS:
        yield {"type": "step", "message": label}
        yield {"type": "cmd", "message": f"$ {cmd}"}
        code, out, err = _run_command(client, cmd)
        combined = (out + err).strip()
        if combined:
            # Emit last 5 lines to keep stream readable
            tail = "\n".join(combined.splitlines()[-5:])
            yield {"type": "output", "message": tail}
        if code != 0:
            client.close()
            yield {"type": "error", "message": f"Step failed (exit {code}): {label}"}
            return
        yield {"type": "ok", "message": f"✓ {label}"}

    client.close()
    yield {"type": "done", "message": "GPU setup complete — all KVForge dependencies installed"}
