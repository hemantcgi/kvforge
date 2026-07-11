"""
Qdrant RAG client — index a PDF and search it with vector similarity.

Usage:
  python3 bedrock_rag.py index <pdf_file>  [options]
  python3 bedrock_rag.py search <query>    [options]
  python3 bedrock_rag.py index  <pdf_file> --config config.json
  python3 bedrock_rag.py search <query>    --config config.json

Config precedence: CLI flags > --config JSON file > built-in defaults.

Config JSON keys (all optional):
  {
    "collection":    "my-collection",
    "qdrant_host":   "localhost",
    "qdrant_port":   6333,
    "embed_model":   "BAAI/bge-small-en-v1.5",
    "vector_dim":    384,
    "chunk_size":    600,
    "chunk_overlap": 60,
    "embed_batch":   64,
    "upsert_batch":  128,
    "top_k":         5
  }
"""

import argparse
import json
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from fastembed import TextEmbedding

from vectorstore.base import Point
from vectorstore.registry import get_store


# ── Configuration dataclass ────────────────────────────────────────────────────

@dataclass
class Config:
    collection:    str = "my-collection"
    qdrant_host:   str = "localhost"
    qdrant_port:   int = 6333
    embed_model:   str = "BAAI/bge-small-en-v1.5"
    vector_dim:    int = 384
    chunk_size:    int = 600
    chunk_overlap: int = 60
    embed_batch:   int = 64
    upsert_batch:  int = 128
    top_k:         int = 5
    score_threshold: float = 0.55
    loader:        str = "pdf"
    vector_store:  str = "qdrant"


def _load_config(args: argparse.Namespace) -> Config:
    """
    Build Config with precedence: defaults → JSON file → explicit CLI flags.
    """
    cfg = Config()

    # Layer 1: JSON config file
    if args.config:
        path = Path(args.config)
        if not path.exists():
            print(f"❌ Config file not found: {path}")
            sys.exit(1)
        with open(path) as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                print(f"⚠️  Unknown config key ignored: '{key}'")

    # Layer 2: explicit CLI flags (only apply if user set them, i.e. not None)
    cli_map = {
        "collection":    args.collection,
        "qdrant_host":   args.qdrant_host,
        "qdrant_port":   args.qdrant_port,
        "embed_model":   args.embed_model,
        "vector_dim":    args.vector_dim,
        "chunk_size":    args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embed_batch":   args.embed_batch,
        "upsert_batch":  args.upsert_batch,
        "top_k":         args.top_k,
    }
    for key, value in cli_map.items():
        if value is not None:
            setattr(cfg, key, value)

    return cfg


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)


def progress(done: int, total: int, label: str = "") -> None:
    pct = done / total * 100
    bar_len = 30
    filled = int(bar_len * done / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r   {label}[{bar}] {done}/{total} ({pct:.0f}%)", end="", flush=True)
    if done == total:
        print()


# ── PDF Reading ────────────────────────────────────────────────────────────────

def read_pdf(path: Path) -> list[dict]:
    """Extract text from each page, return list of {page, text} dicts."""
    from pypdf import PdfReader  # lazy: only needed for PDF ingestion paths
    log(f"📄 Reading PDF: {path.name}")
    reader = PdfReader(str(path))
    total = len(reader.pages)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i, "text": text})
        if i % 200 == 0 or i == total:
            progress(i, total, "Pages ")
    log(f"   → {len(pages)} pages with content")
    return pages


# ── Text Chunking ──────────────────────────────────────────────────────────────

def chunk_pages(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Split each page's text into overlapping word-based chunks."""
    chunks = []
    chunk_id = 0
    step = max(chunk_size - overlap, 1)
    for page_data in pages:
        words = page_data["text"].split()
        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]
            if len(chunk_words) < 30:
                continue
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page_data["page"],
                    "text": " ".join(chunk_words),
                }
            )
            chunk_id += 1
    return chunks


# ── Batched embedding ──────────────────────────────────────────────────────────

def embed_chunks(
    chunks: list[dict],
    embedder: TextEmbedding,
    batch_size: int,
) -> list[list[float]]:
    """Embed all chunks in batches, printing progress to stdout."""
    all_vectors: list[list[float]] = []
    total = len(chunks)
    log(f"🔢 Embedding {total} chunks in batches of {batch_size} …")
    t0 = time.time()

    for start in range(0, total, batch_size):
        batch_texts = [c["text"] for c in chunks[start : start + batch_size]]
        batch_vecs = [v.tolist() for v in embedder.embed(batch_texts)]
        all_vectors.extend(batch_vecs)
        progress(min(start + batch_size, total), total, "Embed ")

    elapsed = time.time() - t0
    log(f"   → Done in {elapsed:.1f}s  ({total / elapsed:.1f} chunks/s)")
    return all_vectors


def validate_embed_dim(embedder, cfg) -> None:
    """Fail fast if embedder output dim doesn't match cfg.vector_dim."""
    test_vec = next(iter(embedder.embed(["dimension check"])))
    actual = len(test_vec)
    if actual != cfg.vector_dim:
        raise ValueError(
            f"Embedding model '{cfg.embed_model}' produces {actual}-dim vectors "
            f"but config declares vector_dim={cfg.vector_dim}. "
            f"Update vector_dim in your datasource config."
        )


# ── Qdrant Indexing ────────────────────────────────────────────────────────────

def index_chunks(
    chunks: list[dict],
    vectors: list[list[float]],
    store,
    cfg: Config,
) -> None:
    """Upsert embedded chunks into the vector store."""
    if store.collection_exists(cfg.collection):
        log(f"🗑️  Deleting existing collection '{cfg.collection}'")
        store.delete_collection(cfg.collection)

    log(f"📦 Creating collection '{cfg.collection}' (dim={cfg.vector_dim}, Cosine)")
    store.create_collection(cfg.collection, cfg.vector_dim)

    total = len(chunks)
    log(f"⬆️  Upserting {total} points …")
    for start in range(0, total, cfg.upsert_batch):
        batch_chunks = chunks[start : start + cfg.upsert_batch]
        batch_vecs   = vectors[start : start + cfg.upsert_batch]
        points = [
            Point(
                id=c["chunk_id"],
                vector=v,
                payload={"page": c["page"], "text": c["text"]},
            )
            for c, v in zip(batch_chunks, batch_vecs)
        ]
        store.upsert(cfg.collection, points)
        progress(min(start + cfg.upsert_batch, total), total, "Upsert ")

    log(f"✅ Indexing complete — {total} vectors stored in '{cfg.collection}'")


# ── Querying ───────────────────────────────────────────────────────────────────

def _run_search(
    question: str,
    embedder: TextEmbedding,
    store,
    cfg: Config,
) -> list:
    """Embed question and return scored results from the vector store."""
    q_vector = next(iter(embedder.embed([question]))).tolist()
    return store.query(cfg.collection, q_vector, cfg.top_k,
                       score_threshold=cfg.score_threshold)


def _emit_json(question: str, results: list) -> None:
    """Emit search results as a single JSON object to stdout (pipe mode)."""
    payload = {
        "query": question,
        "chunks": [
            {
                "page": hit.payload["page"],
                "score": round(hit.score, 4),
                "text": hit.payload["text"],
            }
            for hit in results
        ],
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def query(question: str, embedder: TextEmbedding, store, cfg: Config) -> None:
    """Embed the question, search vector store, and print the top-k results."""
    sep = "─" * 70
    log(f"\n{sep}")
    log(f"❓ Query: {question}")
    log(sep)

    results = _run_search(question, embedder, store, cfg)

    if not results:
        log("⚠️  No results found.")
        return

    log(f"\n🔍 Top {cfg.top_k} most relevant passages:\n")
    for rank, hit in enumerate(results, start=1):
        wrapped = textwrap.fill(
            hit.payload["text"], width=80,
            initial_indent="   ", subsequent_indent="   ",
        )
        log(f"  [{rank}] Score: {hit.score:.4f}  |  Page: {hit.payload['page']}")
        log(wrapped)
        log("")

    log(sep)
    log("💡 Answer — key sentences from the most relevant passages:\n")

    combined = " ".join(hit.payload["text"] for hit in results[:3])
    sentences = [
        s.strip() for s in combined.replace("\n", " ").split(". ")
        if len(s.strip()) > 50
    ]
    q_words = {w.lower().strip("?.,!") for w in question.split() if len(w) > 3}
    relevant = [
        s for s in sentences
        if sum(1 for kw in q_words if kw in s.lower()) >= 2
    ]

    if relevant:
        for sent in relevant[:8]:
            log(f"   • {sent.strip()}.")
    else:
        log(textwrap.fill(
            results[0].payload["text"], width=80,
            initial_indent="   ", subsequent_indent="   ",
        ))


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_index(pdf_path: Path, cfg: Config) -> None:
    if not pdf_path.exists():
        log(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)

    log(f"⚙️  Config: collection={cfg.collection}, host={cfg.qdrant_host}:{cfg.qdrant_port}, "
        f"model={cfg.embed_model}, chunk={cfg.chunk_size}/{cfg.chunk_overlap}")

    from ingestion.registry import get_loader
    loader = get_loader(vars(cfg))
    docs = loader.load(str(pdf_path))
    # Convert to internal chunk format for backwards compatibility
    chunks = [
        {
            "chunk_id": d["metadata"]["chunk_id"],
            "page": d["metadata"].get("page", 0),
            "text": d["text"],
        }
        for d in docs
    ]
    log(f"✂️  Created {len(chunks)} chunks (size≈{cfg.chunk_size} words, overlap={cfg.chunk_overlap})")

    log(f"\n🤖 Loading embedding model '{cfg.embed_model}' …")
    embedder = TextEmbedding(model_name=cfg.embed_model, show_download_progress=False)
    validate_embed_dim(embedder, cfg)

    store = get_store(vars(cfg))
    log(f"🔗 Connecting to vector store ({cfg.vector_store}) …")

    vectors = embed_chunks(chunks, embedder, cfg.embed_batch)
    index_chunks(chunks, vectors, store, cfg)


def cmd_search(question: str, cfg: Config) -> None:
    piped = not sys.stdout.isatty()

    def info(msg: str) -> None:
        if piped:
            print(msg, file=sys.stderr, flush=True)
        else:
            log(msg)

    info(f"🤖 Loading embedding model '{cfg.embed_model}' …")
    embedder = TextEmbedding(model_name=cfg.embed_model, show_download_progress=False)

    store = get_store(vars(cfg))
    info(f"🔗 Connecting to vector store ({cfg.vector_store}) …")

    if not store.collection_exists(cfg.collection):
        log(f"❌ Collection '{cfg.collection}' not found. Run 'index' first.")
        sys.exit(1)

    if piped:
        results = _run_search(question, embedder, store, cfg)
        _emit_json(question, results)
    else:
        query(question, embedder, store, cfg)


# ── Argument parser ────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bedrock_rag.py",
        description="Index a PDF into Qdrant and search it with vector similarity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Shared options (global flags before the subcommand) ───────────────────
    parser.add_argument("--config",        metavar="FILE",  default=None,
                        help="JSON config file (keys override defaults; CLI flags override file)")
    parser.add_argument("--collection",    metavar="NAME",  default=None,
                        help=f"Qdrant collection name (default: {Config.collection})")
    parser.add_argument("--qdrant-host",   metavar="HOST",  default=None,
                        help=f"Qdrant host (default: {Config.qdrant_host})")
    parser.add_argument("--qdrant-port",   metavar="PORT",  type=int, default=None,
                        help=f"Qdrant REST port (default: {Config.qdrant_port})")
    parser.add_argument("--embed-model",   metavar="MODEL", default=None,
                        help=f"fastembed model name (default: {Config.embed_model})")
    parser.add_argument("--vector-dim",    metavar="N",     type=int, default=None,
                        help=f"Embedding dimension — must match model (default: {Config.vector_dim})")
    parser.add_argument("--top-k",         metavar="K",     type=int, default=None,
                        help=f"Number of results to retrieve (default: {Config.top_k})")

    # ── Index-only options ────────────────────────────────────────────────────
    parser.add_argument("--chunk-size",    metavar="N",     type=int, default=None,
                        help=f"Words per chunk (default: {Config.chunk_size})")
    parser.add_argument("--chunk-overlap", metavar="N",     type=int, default=None,
                        help=f"Overlap between chunks in words (default: {Config.chunk_overlap})")
    parser.add_argument("--embed-batch",   metavar="N",     type=int, default=None,
                        help=f"Embedding batch size (default: {Config.embed_batch})")
    parser.add_argument("--upsert-batch",  metavar="N",     type=int, default=None,
                        help=f"Qdrant upsert batch size (default: {Config.upsert_batch})")

    # ── Subcommands ───────────────────────────────────────────────────────────
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    idx = sub.add_parser("index",  help="Embed and index a PDF file")
    idx.add_argument("pdf_file", metavar="PDF_FILE", help="Path to the PDF to index")

    srch = sub.add_parser("search", help="Search the vector database")
    srch.add_argument("query", nargs="+", metavar="WORD",
                      help="Search query (all words are joined into one query string)")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # argparse uses underscores; map hyphenated dest names
    args.qdrant_host   = getattr(args, "qdrant_host",   None)
    args.qdrant_port   = getattr(args, "qdrant_port",   None)
    args.embed_model   = getattr(args, "embed_model",   None)
    args.vector_dim    = getattr(args, "vector_dim",    None)
    args.chunk_size    = getattr(args, "chunk_size",    None)
    args.chunk_overlap = getattr(args, "chunk_overlap", None)
    args.embed_batch   = getattr(args, "embed_batch",   None)
    args.upsert_batch  = getattr(args, "upsert_batch",  None)
    args.top_k         = getattr(args, "top_k",         None)

    cfg = _load_config(args)

    if args.command == "index":
        cmd_index(Path(args.pdf_file), cfg)
    elif args.command == "search":
        cmd_search(" ".join(args.query), cfg)


if __name__ == "__main__":
    main()
