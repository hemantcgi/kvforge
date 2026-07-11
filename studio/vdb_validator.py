# studio/vdb_validator.py
import sys
from pathlib import Path
import requests


def validate(config: dict) -> dict:
    dispatch = {
        "qdrant":   _validate_qdrant,
        "chroma":   _validate_chroma,
        "faiss":    _validate_faiss,
        "pinecone": _validate_pinecone,
        "weaviate": _validate_weaviate,
        "milvus":   _validate_milvus,
        "generic":  _validate_generic,
    }
    fn = dispatch.get(config.get("type", ""))
    if fn is None:
        return {"ok": False, "error": f"Unknown VDB type: {config.get('type')}", "collection_count": None}
    try:
        return fn(config)
    except Exception as e:
        return {"ok": False, "error": str(e), "collection_count": None}


def _validate_qdrant(config: dict) -> dict:
    if "qdrant_client" not in sys.modules or sys.modules.get("qdrant_client") is None:
        try:
            import qdrant_client  # noqa: F401
        except (ImportError, TypeError):
            return {"ok": False, "error": "qdrant-client not installed — pip install qdrant-client", "collection_count": None}
    from qdrant_client import QdrantClient
    client = QdrantClient(
        host=config.get("host", "localhost"),
        port=int(config.get("port", 6333)),
        api_key=config.get("api_key") or None,
        timeout=5,
    )
    cols = client.get_collections().collections
    return {"ok": True, "error": None, "collection_count": len(cols)}


def _validate_chroma(config: dict) -> dict:
    try:
        import chromadb
    except ImportError:
        return {"ok": False, "error": "chromadb not installed — pip install chromadb", "collection_count": None}
    client = chromadb.HttpClient(host=config.get("host", "localhost"), port=int(config.get("port", 8000)))
    cols = client.list_collections()
    return {"ok": True, "error": None, "collection_count": len(cols)}


def _validate_faiss(config: dict) -> dict:
    p = Path(config.get("index_path", ""))
    if not p.exists():
        return {"ok": False, "error": f"Index file not found: {p}", "collection_count": None}
    return {"ok": True, "error": None, "collection_count": 1}


def _validate_pinecone(config: dict) -> dict:
    try:
        from pinecone import Pinecone
    except ImportError:
        return {"ok": False, "error": "pinecone-client not installed — pip install pinecone-client", "collection_count": None}
    pc = Pinecone(api_key=config.get("api_key", ""))
    idxs = pc.list_indexes()
    return {"ok": True, "error": None, "collection_count": len(idxs)}


def _validate_weaviate(config: dict) -> dict:
    url = config.get("url", "").rstrip("/")
    api_key = config.get("api_key")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(f"{url}/v1/.well-known/ready", headers=headers, timeout=5)
    if resp.status_code != 200:
        return {"ok": False, "error": f"Weaviate not ready: HTTP {resp.status_code}", "collection_count": None}
    return {"ok": True, "error": None, "collection_count": None}


def _validate_milvus(config: dict) -> dict:
    try:
        from pymilvus import MilvusClient
    except ImportError:
        return {"ok": False, "error": "pymilvus not installed — pip install pymilvus", "collection_count": None}
    uri = f"http://{config.get('host', 'localhost')}:{config.get('port', 19530)}"
    client = MilvusClient(uri=uri, token=config.get("token", ""))
    cols = client.list_collections()
    return {"ok": True, "error": None, "collection_count": len(cols)}


def _validate_generic(config: dict) -> dict:
    url = config.get("base_url", "")
    key = config.get("auth_header_key", "")
    val = config.get("auth_header_value", "")
    headers = {key: val} if key else {}
    resp = requests.get(url, headers=headers, timeout=5)
    if resp.status_code >= 500:
        return {"ok": False, "error": f"Endpoint returned HTTP {resp.status_code}", "collection_count": None}
    return {"ok": True, "error": None, "collection_count": None}
