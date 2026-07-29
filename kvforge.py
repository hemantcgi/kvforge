"""KVForge command-line interface.

Provides subcommands to initialise, index, and search a KVForge datasource.

Commands:

* ``init``   — scaffold a new datasource config JSON file.
* ``index``  — load documents, embed them, and upsert into the vector store.
* ``search`` — embed a query and print the top-K results.

Usage::

    python kvforge.py init   --name my-corpus
    python kvforge.py index  --config datasource_my-corpus.json --source ./docs/
    python kvforge.py search --config datasource_my-corpus.json "my query"
"""

import argparse
import json
import os
import sys
from pathlib import Path


def cmd_init(args) -> None:
    """Scaffold a new KVForgeConfig JSON file with sensible defaults.

    Writes the new addon-based config to ``datasource_<name>.json``.
    Exits with an error message if the file already exists and ``--force``
    is not set.

    Args:
        args: Parsed argument namespace.  Expected attributes: ``name``,
            ``loader``, ``embed_model``, ``vector_dim``, ``llm_model``,
            ``force``.
    """
    name = args.name
    safe = name.replace(" ", "-").lower()
    config_path = f"datasource_{safe}.json"
    if Path(config_path).exists() and not args.force:
        print(f"Config already exists: {config_path}. Use --force to overwrite.")
        sys.exit(1)

    template = {
        "_comment": f"KVForge config for '{name}' — edit addon_config before running 'kvforge start'",
        "use_case_name": name,
        "collection": safe,
        "version_file": f"{safe}/version.json",
        "addons": ["indexing", "inference", "training", "background", "monitoring"],
        "addon_config": {
            "indexing": {
                "loader": args.loader,
                "chunk_size": 600,
                "chunk_overlap": 60,
                "embed_batch": 64,
                "upsert_batch": 128,
                "embed_model": args.embed_model,
                "embedder_backend": "fastembed",
                "vector_dim": args.vector_dim,
                "vector_store": "qdrant",
                "qdrant_host": "localhost",
                "qdrant_port": 6333,
            },
            "inference": {
                "top_k": 5,
                "llm_model": args.llm_model,
                "quantization": "4bit",
                "vllm_url": "http://localhost:8091",
                "max_new_tokens": 256,
                "gate_threshold": 0.75,
            },
            "training": {
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_target_modules": ["q_proj", "k_proj", "v_proj"],
                "lora_dropout": 0.05,
                "lora_epochs": 3,
                "lora_lr": 0.0002,
                "checkpoint_dir": f"{safe}/lora_checkpoints/",
                "replay_db": f"{safe}/replay.db",
                "prs_threshold": 0.50,
                "faq_question_key": "question",
                "faq_answer_key": "answer",
            },
            "background": {"flush_seconds": 300, "flush_queries": 50},
            "monitoring": {"port": 8082},
        },
    }

    os.makedirs(safe, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(template, f, indent=2)

    print(f"Created {config_path}")
    print(f"Next: edit addon_config fields, then run: kvforge start --config {config_path}")


def cmd_start(args) -> None:
    """Launch the per-use-case KVForge Dashboard.

    Args:
        args: Parsed argument namespace.  Expected attributes: ``config``,
            ``port``.
    """
    import uvicorn
    from dashboard.app import create_app

    port = args.port
    if Path(args.config).exists():
        try:
            from core.config import load_config
            cfg = load_config(args.config)
            if cfg.has_addon("monitoring"):
                port = cfg.addon_config.get("monitoring", {}).get("port", port)
        except Exception:
            pass  # fall back to --port arg if config is malformed

    app = create_app(config_path=args.config)
    print(f"KVForge Dashboard starting at http://localhost:{port}")
    print(f"Config: {args.config}")
    uvicorn.run(app, host="0.0.0.0", port=port)


def cmd_index(args) -> None:
    """Load, embed, and upsert documents from *args.source* into the vector store.

    Deletes and recreates the collection before upserting so that re-indexing
    always produces a clean state.  Progress is printed to stdout.

    Args:
        args: Parsed argument namespace.  Expected attributes: ``config``,
            ``source``.
    """
    import json as _json
    from ingestion.registry import get_loader
    from embeddings.registry import get_embedder
    from vectorstore.registry import get_store
    from vectorstore.base import Point

    with open(args.config) as f:
        cfg = _json.load(f)

    # Flatten nested addon_config for legacy pipeline code.
    addon_config = cfg.get("addon_config", {})
    for section in ("indexing", "inference", "training", "background", "sync", "monitoring"):
        cfg.update(addon_config.get(section, {}))

    loader = get_loader(cfg)
    embedder = get_embedder(cfg)
    store = get_store(cfg)

    print(f"Loading documents from {args.source}...")
    docs = loader.load(args.source)
    print(f"Loaded {len(docs)} chunks")

    texts = [d["text"] for d in docs]
    print(f"Embedding {len(texts)} chunks...")
    vectors = embedder.encode(texts)

    collection = cfg["collection"]
    if store.collection_exists(collection):
        store.delete_collection(collection)
    store.create_collection(collection, embedder.dim)

    points = [Point(id=i, vector=v, payload={**d["metadata"], "text": d["text"]})
               for i, (d, v) in enumerate(zip(docs, vectors))]
    batch = cfg.get("upsert_batch", 128)
    for start in range(0, len(points), batch):
        store.upsert(collection, points[start:start + batch])
        print(f"  Upserted {min(start + batch, len(points))}/{len(points)}", end="\r")
    print(f"\nIndexed {len(points)} points into '{collection}'")


def cmd_search(args) -> None:
    """Embed *args.query* and print the top-K matching chunks from the collection.

    Args:
        args: Parsed argument namespace.  Expected attributes: ``config``,
            ``query``.
    """
    import json as _json
    from embeddings.registry import get_embedder
    from vectorstore.registry import get_store

    with open(args.config) as f:
        cfg = _json.load(f)

    # Flatten nested addon_config for legacy pipeline code.
    addon_config = cfg.get("addon_config", {})
    for section in ("indexing", "inference", "training", "background", "sync", "monitoring"):
        cfg.update(addon_config.get(section, {}))

    embedder = get_embedder(cfg)
    store = get_store(cfg)
    vector = embedder.encode([args.query])[0]
    results = store.query(cfg["collection"], vector, top_k=cfg.get("top_k", 5))
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r.score:.4f}")
        print(r.payload.get("text", "")[:300])


def main() -> None:
    parser = argparse.ArgumentParser(prog="kvforge", description="KVForge CLI")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # init
    p_init = sub.add_parser("init", help="Create a new datasource config")
    p_init.add_argument("--name", required=True)
    p_init.add_argument("--loader", default="pdf",
                         choices=["pdf", "markdown", "jsonl", "html", "directory"])
    p_init.add_argument("--embed-model", dest="embed_model",
                         default="BAAI/bge-small-en-v1.5")
    p_init.add_argument("--vector-dim", dest="vector_dim", type=int, default=384)
    p_init.add_argument("--llm-model", dest="llm_model",
                         default="meta-llama/Llama-3.2-3B-Instruct")
    p_init.add_argument("--force", action="store_true")

    # start
    p_start = sub.add_parser("start", help="Launch the per-use-case KVForge Dashboard")
    p_start.add_argument("--config", default="config.json",
                         help="Path to KVForgeConfig JSON (default: config.json)")
    p_start.add_argument("--port", type=int, default=8080,
                         help="Dashboard port (default: 8080; overridden by monitoring.port in config)")

    # index
    p_idx = sub.add_parser("index", help="Index a source into the collection")
    p_idx.add_argument("--config", required=True)
    p_idx.add_argument("--source", required=True)

    # search
    p_srch = sub.add_parser("search", help="Search the collection")
    p_srch.add_argument("--config", required=True)
    p_srch.add_argument("query")

    args = parser.parse_args()
    if args.command == "init":
        cmd_init(args)
    elif args.command == "start":
        cmd_start(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()
