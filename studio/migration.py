# studio/migration.py
"""One-time migration: creates kvforge_registry.json and uc_config.json from config.json."""

import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent

_STORE_MAP = {"qdrant": "qdrant", "chromadb": "chromadb", "faiss": "faiss"}
_DISPLAY_NAMES = {
    "usecase1_customer_support": "Customer Support",
    "usecase2_pubmedqa":          "PubMedQA",
    "usecase3_squad":             "SQuAD",
    "usecase4_bedrock_userguide": "Bedrock User Guide",
}


def _config_to_uc_config(uc_id: str, cfg: dict) -> dict:
    loader = cfg.get("loader", "jsonl")
    source_type = "pdf" if loader == "pdf" else "huggingface"
    return {
        "id":           uc_id,
        "display_name": _DISPLAY_NAMES.get(uc_id, uc_id),
        "type":         "example",
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "data": {
            "source_type":  source_type,
            "source_path":  "examples/usecase4_bedrock_userguide/data/" if loader == "pdf" else "",
            "dataset_id":   cfg.get("dataset_id", ""),
            "split":        cfg.get("split", "train"),
            "text_column":  cfg.get("jsonl_text_key", "text"),
            "max_rows":     cfg.get("max_rows", 5000),
        },
        "vectordb": {
            "store":           _STORE_MAP.get(cfg.get("vector_store", "qdrant"), "qdrant"),
            "dimensions":      cfg.get("vector_dim", 384),
            "chunk_size":      cfg.get("chunk_size", 512),
            "chunk_overlap":   cfg.get("chunk_overlap", 64),
            "embedding_model": cfg.get("embed_model", "BAAI/bge-small-en-v1.5"),
            "index_type":      "hnsw",
        },
        "llm": {
            "local_model":          cfg.get("llm_model", "meta-llama/Llama-3.2-3B-Instruct"),
            "quantization":         cfg.get("quantization", "4bit"),
            "vllm_url":             cfg.get("vllm_url", ""),
            "comparison_provider":  "gemini",
            "comparison_model":     "gemini-1.5-flash",
        },
    }


def migrate_existing_use_cases(root: Path = ROOT):
    registry_path = root / "kvforge_registry.json"

    # Load existing registry or start fresh
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    else:
        registry = {"use_cases": []}

    existing_ids = {uc["id"] for uc in registry["use_cases"]}

    for config_path in sorted((root / "examples").glob("*/config.json")):
        uc_id = config_path.parent.name
        cfg   = json.loads(config_path.read_text())

        # Write uc_config.json (overwrite to keep fresh)
        uc_config = _config_to_uc_config(uc_id, cfg)
        uc_config_path = config_path.parent / "uc_config.json"
        uc_config_path.write_text(json.dumps(uc_config, indent=2))

        # Add to registry if not already present
        if uc_id not in existing_ids:
            registry["use_cases"].append({
                "id":           uc_id,
                "display_name": uc_config["display_name"],
                "type":         "example",
            })
            existing_ids.add(uc_id)

    registry_path.write_text(json.dumps(registry, indent=2))


def load_registry(root: Path = ROOT) -> list[dict]:
    registry_path = root / "kvforge_registry.json"
    if not registry_path.exists():
        return []
    return json.loads(registry_path.read_text()).get("use_cases", [])


def add_to_registry(uc_id: str, display_name: str, root: Path = ROOT):
    registry_path = root / "kvforge_registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"use_cases": []}
    if not any(uc["id"] == uc_id for uc in registry["use_cases"]):
        registry["use_cases"].append({"id": uc_id, "display_name": display_name, "type": "custom"})
    registry_path.write_text(json.dumps(registry, indent=2))
