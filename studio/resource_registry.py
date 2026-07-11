"""Resource provider registry for KVForge Studio.

Stores typed resource providers (VectorDB, LLM endpoints, Cloud LLMs, Compute)
in ~/.kvforge/resource_providers.json. Each provider has a unique ID, type,
backend, display name, and config dict.
"""

import json
import uuid
from pathlib import Path

_STORE = Path.home() / ".kvforge" / "resource_providers.json"

PROVIDER_TYPES = ("vectordb", "llm_endpoint", "cloud_llm", "compute")

BACKENDS = {
    "vectordb":    ("qdrant", "chroma", "faiss"),
    "llm_endpoint": ("vllm", "ollama"),
    "cloud_llm":   ("anthropic", "openai", "gemini"),
    "compute":     ("ssh_gpu", "local_gpu"),
}


_GPU_PROFILES = Path.home() / ".kvforge" / "gpu_profiles.json"
_SEEDED = False


def _load() -> list[dict]:
    if not _STORE.exists():
        return []
    try:
        return json.loads(_STORE.read_text())
    except Exception:
        return []


def _save(providers: list[dict]) -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    _STORE.write_text(json.dumps(providers, indent=2))


def _seed_from_existing() -> None:
    """Import GPU profiles, local Qdrant, and configured cloud API keys on first run."""
    global _SEEDED
    if _SEEDED:
        return
    _SEEDED = True

    providers = _load()
    existing_keys: set[str] = set()
    for p in providers:
        b, c = p.get("backend", ""), p.get("config", {})
        if b == "ssh_gpu":
            existing_keys.add(f"ssh:{c.get('host')}:{c.get('port', 22)}")
        elif b == "qdrant":
            existing_keys.add(f"qdrant:{c.get('host', 'localhost')}:{c.get('port', 6333)}")
        elif b in ("anthropic", "openai", "gemini"):
            existing_keys.add(f"cloud:{b}")
        elif b == "local_gpu":
            existing_keys.add("local_gpu")

    changed = False

    # 1. Import SSH GPU profiles
    if _GPU_PROFILES.exists():
        try:
            gpu_profiles = json.loads(_GPU_PROFILES.read_text())
            for gp in gpu_profiles:
                key = f"ssh:{gp.get('host')}:{gp.get('port', 22)}"
                if key not in existing_keys:
                    providers.append({
                        "id": str(uuid.uuid4())[:8],
                        "type": "compute",
                        "backend": "ssh_gpu",
                        "display_name": gp.get("display_name") or f"GPU @ {gp['host']}",
                        "config": {
                            "host": gp["host"],
                            "port": gp.get("port", 22),
                            "user": gp.get("user", "ubuntu"),
                            "pem_enc": gp.get("pem_enc", ""),
                            "fingerprint": gp.get("fingerprint", ""),
                        },
                    })
                    existing_keys.add(key)
                    changed = True
        except Exception:
            pass

    # 2. Add local Qdrant at localhost:6333 if not present
    if "qdrant:localhost:6333" not in existing_keys:
        providers.append({
            "id": str(uuid.uuid4())[:8],
            "type": "vectordb",
            "backend": "qdrant",
            "display_name": "Local Qdrant",
            "config": {"host": "localhost", "port": 6333},
        })
        existing_keys.add("qdrant:localhost:6333")
        changed = True

    # 3. Add cloud LLMs for configured API keys
    try:
        from studio.settings_manager import get_setting
        cloud_map = {
            "anthropic": ("anthropic_api_key", "claude-haiku-4-5-20251001"),
            "openai":    ("openai_api_key",    "gpt-4o-mini"),
            "gemini":    ("gemini_api_key",    "gemini-2.5-flash"),
        }
        for backend, (setting_key, default_model) in cloud_map.items():
            key = f"cloud:{backend}"
            if key not in existing_keys and get_setting(setting_key):
                providers.append({
                    "id": str(uuid.uuid4())[:8],
                    "type": "cloud_llm",
                    "backend": backend,
                    "display_name": {"anthropic": "Anthropic (Claude)", "openai": "OpenAI", "gemini": "Google Gemini"}[backend],
                    "config": {"default_model": default_model},
                })
                existing_keys.add(key)
                changed = True
    except Exception:
        pass

    if changed:
        _save(providers)


def list_providers(provider_type: str | None = None) -> list[dict]:
    _seed_from_existing()
    providers = _load()
    if provider_type:
        providers = [p for p in providers if p.get("type") == provider_type]
    return [{k: v for k, v in p.items() if k != "config_secret"} for p in providers]


def get_provider(provider_id: str) -> dict | None:
    for p in _load():
        if p["id"] == provider_id:
            return p
    return None


def create_provider(
    provider_type: str,
    backend: str,
    display_name: str,
    config: dict,
) -> dict:
    if provider_type not in PROVIDER_TYPES:
        raise ValueError(f"Unknown provider type: {provider_type!r}")
    providers = _load()
    record = {
        "id": str(uuid.uuid4())[:8],
        "type": provider_type,
        "backend": backend,
        "display_name": display_name,
        "config": config,
    }
    providers.append(record)
    _save(providers)
    return record


def update_provider(provider_id: str, **kwargs) -> dict:
    providers = _load()
    for p in providers:
        if p["id"] == provider_id:
            for k, v in kwargs.items():
                if k in ("display_name", "config", "backend"):
                    p[k] = v
            _save(providers)
            return p
    raise KeyError(provider_id)


def delete_provider(provider_id: str) -> None:
    providers = [p for p in _load() if p["id"] != provider_id]
    _save(providers)


def test_provider(provider_id: str) -> dict:
    """Quick connectivity check for a registered provider."""
    p = get_provider(provider_id)
    if p is None:
        return {"ok": False, "error": "Provider not found"}
    ptype = p.get("type")
    backend = p.get("backend")
    cfg = p.get("config", {})

    try:
        if ptype == "vectordb" and backend == "qdrant":
            import httpx
            host = cfg.get("host", "localhost")
            port = cfg.get("port", 6333)
            r = httpx.get(f"http://{host}:{port}/collections", timeout=5)
            count = len(r.json().get("result", {}).get("collections", []))
            return {"ok": True, "detail": f"Qdrant reachable — {count} collections"}

        elif ptype == "llm_endpoint" and backend == "vllm":
            import httpx
            url = cfg.get("endpoint_url", "").rstrip("/")
            if not url:
                return {"ok": False, "error": "No endpoint_url configured"}
            base = url[:-3] if url.endswith("/v1") else url
            r = httpx.get(f"{base}/v1/models", timeout=5)
            models = r.json().get("data", [])
            names = [m.get("id", "?") for m in models]
            return {"ok": True, "detail": f"vLLM reachable — models: {', '.join(names) or 'none'}"}

        elif ptype == "llm_endpoint" and backend == "ollama":
            import httpx
            url = cfg.get("endpoint_url", "http://localhost:11434").rstrip("/")
            r = httpx.get(f"{url}/api/tags", timeout=5)
            models = r.json().get("models", [])
            return {"ok": True, "detail": f"Ollama reachable — {len(models)} models"}

        elif ptype == "cloud_llm":
            # Cloud LLMs are always "reachable" if key is set — just confirm key presence
            from studio.settings_manager import get_setting
            key_map = {"anthropic": "anthropic_api_key", "openai": "openai_api_key", "gemini": "gemini_api_key"}
            setting_key = key_map.get(backend, "")
            api_key = cfg.get("api_key") or (get_setting(setting_key) if setting_key else "")
            if api_key:
                return {"ok": True, "detail": f"API key configured for {backend}"}
            return {"ok": False, "error": f"No API key found — set it in Settings → API Keys"}

        elif ptype == "compute" and backend == "ssh_gpu":
            from studio.remote_gpu import _make_client, _run_command, _decrypt
            # ssh_gpu providers seeded from gpu_profiles carry pem_enc (encrypted)
            # or pem_content (plain, from manual add modal)
            pem = cfg.get("pem_content") or ""
            if not pem and cfg.get("pem_enc"):
                pem = _decrypt(cfg["pem_enc"])
            if not pem:
                return {"ok": False, "error": "No PEM key stored for this provider"}
            client = _make_client(cfg["host"], cfg.get("user", "ubuntu"), cfg.get("port", 22), pem)
            code, out, _ = _run_command(client, "nvidia-smi --query-gpu=name --format=csv,noheader")
            client.close()
            if code != 0:
                return {"ok": False, "error": "SSH connected but nvidia-smi not found"}
            gpus = out.strip().splitlines()
            return {"ok": True, "detail": f"SSH OK — GPUs: {', '.join(gpus) or 'none detected'}"}

        elif ptype == "compute" and backend == "local_gpu":
            try:
                import torch
                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    return {"ok": True, "detail": f"Local GPU: {name}"}
                return {"ok": False, "error": "No CUDA GPU detected locally"}
            except ImportError:
                return {"ok": False, "error": "torch not installed"}

        return {"ok": False, "error": f"No connectivity test for {ptype}/{backend}"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:120]}
