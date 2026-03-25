"""SmartQdrant command-line interface.

Provides subcommands to initialise, index, and search a SmartQdrant datasource.

Commands:

* ``init``   — scaffold a new datasource config JSON file.
* ``index``  — load documents, embed them, and upsert into the vector store.
* ``search`` — embed a query and print the top-K results.

Usage::

    python smartqdrant.py init   --name my-corpus
    python smartqdrant.py index  --config datasource_my-corpus.json --source ./docs/
    python smartqdrant.py search --config datasource_my-corpus.json "my query"
"""

import argparse
import json
import os
import sys
from pathlib import Path


def cmd_init(args) -> None:
    """Scaffold a new datasource config JSON file with sensible defaults.

    Writes a validated ``DatasourceConfig`` JSON to
    ``datasource_<name>.json``.  Exits with an error message if the file
    already exists and ``--force`` is not set.

    Args:
        args: Parsed argument namespace.  Expected attributes: ``name``,
            ``loader``, ``embed_model``, ``vector_dim``, ``llm_model``,
            ``force``.
    """
    name = args.name
    config_path = f"datasource_{name}.json"
    if Path(config_path).exists() and not args.force:
        print(f"Config already exists: {config_path}. Use --force to overwrite.")
        sys.exit(1)

    cfg = {
        "collection": name,
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "vector_store": "qdrant",
        "loader": args.loader,
        "embed_model": args.embed_model,
        "embedder_backend": "fastembed",
        "vector_dim": args.vector_dim,
        "llm_model": args.llm_model,
        "chunk_size": 600,
        "chunk_overlap": 60,
        "embed_batch": 64,
        "upsert_batch": 128,
        "top_k": 5,
        "model_library": {},
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj"],
        "lora_dropout": 0.05,
        "lora_epochs": 3,
        "lora_lr": 0.0002,
        "checkpoint_dir": f"lora_checkpoints/{name}/",
        "version_file": f"{name}_version.json",
        "replay_db": f"{name}_replay.db",
        "gate_threshold": 0.75,
        "prs_threshold": 0.75,
        "prs_weights": {"accuracy": 0.5, "calibration": 0.3, "consistency": 0.2},
        "faq_question_key": "question",
        "faq_answer_key": "answer",
        "access_flush_seconds": 300,
        "access_flush_queries": 50,
        "dashboard_port": 8080,
    }

    # Validate before writing
    from config import DatasourceConfig
    DatasourceConfig(**cfg)  # raises ValidationError if invalid

    # Create checkpoint dir
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Created {config_path}")
    print(f"Next steps:")
    print(f"  1. Index your source:  python smartqdrant.py index --config {config_path} --source <path>")
    print(f"  2. Generate FAQs:      python tools/generate_faqs.py --config {config_path} --output {name}_faqs.json")
    print(f"  3. Train:              python -m pipeline.index_and_train --config {config_path} --source <path> --faqs {name}_faqs.json")


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

    embedder = get_embedder(cfg)
    store = get_store(cfg)
    vector = embedder.encode([args.query])[0]
    results = store.query(cfg["collection"], vector, top_k=cfg.get("top_k", 5))
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r.score:.4f}")
        print(r.payload.get("text", "")[:300])


def main() -> None:
    parser = argparse.ArgumentParser(prog="smartqdrant", description="SmartQdrant CLI")
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
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()
