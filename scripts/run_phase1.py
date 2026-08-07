#!/usr/bin/env python3
"""Phase 1 — Prepare Natural Questions dataset and run absorption curve size sweep.

Run on the remote GPU instance via:
    cd /home/ubuntu/kvforge && nohup ./venv/bin/python3 scripts/run_phase1.py > logs/phase1.log 2>&1 &
"""
import json, os, sys, time, subprocess, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
NQ_DIR = ROOT / "examples" / "natural_questions"
NQ_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAhi-xaDAldgAUa8E7JcBhtzhdBdOg2Nho")
# Hard-coded as backup


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, check=True):
    log("$ " + " ".join(cmd) if isinstance(cmd, list) else f"$ {cmd}")
    if isinstance(cmd, str):
        return subprocess.run(cmd, shell=True, check=check)
    return subprocess.run(cmd, check=check)


# ─── Step 1: Create config ───────────────────────────────────────────────
log("=== Phase 1: Natural Questions Dataset Preparation ===")

if not (NQ_DIR / "config.json").exists():
    log("Creating NQ config...")
    cfg = {
        "use_case_name": "Natural Questions",
        "collection": "natural_questions",
        "version_file": str(NQ_DIR / "version.json"),
        "addons": ["indexing", "inference", "training", "background"],
        "addon_config": {
            "indexing": {
                "loader": "jsonl",
                "embed_model": "BAAI/bge-small-en-v1.5",
                "vector_dim": 384,
                "vector_store": "qdrant",
                "qdrant_host": "localhost",
                "qdrant_port": 6333,
            },
            "inference": {
                "llm_model": "google/gemma-4-E2B-it",
                "top_k": 5,
                "max_new_tokens": 128,
            },
            "training": {
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_lr": 2e-4,
                "lora_epochs": 1,
                "checkpoint_dir": str(NQ_DIR / "lora_checkpoints"),
                "replay_db": str(NQ_DIR / "replay.db"),
                "sft_format": "chat",
            },
            "background": {"flush_seconds": 300, "flush_queries": 50},
        },
        "llm_model": "google/gemma-4-E2B-it",
        "model_library": {
            "google/gemma-4-E2B-it": {"num_layers": 35, "num_kv_heads": 1, "head_dim": 256},
        },
    }
    (NQ_DIR / "config.json").write_text(json.dumps(cfg, indent=2))
    
    # version.json
    ver = {"current_lora_version": 0, "checkpoint_path": None, "phase": 1, "prs_history": []}
    (NQ_DIR / "version.json").write_text(json.dumps(ver, indent=2))
    log("Config created.")
else:
    log("Config already exists.")

# ─── Step 2: Download and chunk NQ ────────────────────────────────────────
if not (NQ_DIR / "corpus.jsonl").exists() or os.stat(NQ_DIR / "corpus.jsonl").st_size < 100000:
    log("Downloading Natural Questions from HuggingFace...")
    run([
        sys.executable, "-c", """
import json, datasets, re
from pathlib import Path

nq = datasets.load_dataset("natural_questions", trust_remote_code=True)
out_path = Path('examples/natural_questions/corpus.jsonl')
chunk_size = 600
overlap = 60

with open(out_path, 'w') as f:
    count = 0
    for split_name in ['train', 'validation']:
        for item in nq[split_name]:
            text = item.get('document', {}).get('text', '') or item.get('text', '')
            if not text:
                continue
            # Simple chunking: split into ~600 char chunks with overlap
            words = text.split()
            chunks = []
            i = 0
            while i * chunk_size < len(' '.join(words)):
                start = i * (chunk_size - overlap)
                end = start + chunk_size
                chunk_text = ' '.join(words[start:end])
                if chunk_text:
                    chunks.append(chunk_text)
                i += 1
            for chunk in chunks[:10]:  # max 10 chunks per doc
                f.write(json.dumps({"text": chunk}) + '\\n')
                count += 1
                if count >= 10000:
                    break
        if count >= 10000:
            break
    print(f"Wrote {count} chunks to {out_path}")
"""
    ])
    log(f"Download complete: {(NQ_DIR / 'corpus.jsonl').stat().st_size} bytes")
else:
    log(f"Corpus exists: {(NQ_DIR / 'corpus.jsonl').stat().st_size} bytes")

# ─── Step 3: Generate held-out questions ────────────────────────────────
if not (NQ_DIR / "eval_heldout.json").exists():
    log("Generating held-out questions (sampling from NQ validation split)...")
    run([
        sys.executable, "-c", """
import json, datasets, random
from pathlib import Path

nq = datasets.load_dataset("natural_questions", trust_remote_code=True)
val = nq["validation"]
items = []
rng = random.Random(42)

# Sample 300 candidate questions
samples = rng.sample(range(len(val)), min(300, len(val)))
for idx in samples:
    item = val[int(idx)]
    q = item.get("question", {}).get("text", "")
    # Get answers
    annotations = item.get("annotations", {})
    answers = annotations.get("short_answers", []) or annotations.get("yes_no_answer", [])
    if not answers and "long_answer" in annotations:
        answers = [annotations["long_answer"]]
    if q and answers:
        items.append({
            "question": q,
            "answer": str(answers[0]) if isinstance(answers, list) else str(answers),
            "source_chunk_ids": [],
        })
    if len(items) >= 200:
        break

# Save
output = {"items": items[:200]}
Path("examples/natural_questions/eval_heldout.json").write_text(json.dumps(output, indent=2))
print(f"Saved {len(items[:200])} held-out questions")
"""
    ])
    log("Held-out questions generated.")
else:
    eval_count = len(json.loads((NQ_DIR / "eval_heldout.json").read_text()).get("items", []))
    log(f"Held-out exists: {eval_count} questions")

# ─── Step 4: Index into Qdrant ─────────────────────────────────────────
index_done = (NQ_DIR / "chunks.json").exists()
if not index_done:
    log("Indexing into Qdrant...")
    run([sys.executable, "-m", "pipeline.kv_indexer",
         "--config", str(NQ_DIR / "config.json"),
         "index", str(NQ_DIR / "corpus.jsonl")])
    log("Indexing complete.")

    # Extract embeddings for corpus_slicer
    log("Extracting embeddings...")
    run([
        sys.executable, "-c", """
import json, numpy as np
from pathlib import Path
from vectorstore.registry import get_store

cfg = json.loads(Path('examples/natural_questions/config.json').read_text())
store = get_store(cfg)
chunks, embs = [], []
results, offset = store.scroll(cfg['collection'], limit=1000, with_payload=True, with_vectors=True)
while results:
    for r in results:
        chunks.append({'chunk_id': r.id, 'text': r.payload['text']})
        if r.vector:
            embs.append(r.vector)
    results, offset = store.scroll(cfg['collection'], limit=1000, with_payload=True, with_vectors=True, offset=offset)
    if offset is None:
        break

np.save('examples/natural_questions/embeddings.npy', np.array(embs))
Path('examples/natural_questions/chunks.json').write_text(json.dumps(chunks))
print(f"Saved {len(chunks)} chunks, {len(embs)} embeddings")
"""
    ])
    log("Embeddings extracted.")
else:
    log(f"Index data exists: {os.stat(NQ_DIR / 'chunks.json').st_size} bytes")

# ─── Step 5: Create size tiers ──────────────────────────────────────────
log("=== Phase 1: Size Sweep ===")
from tools.corpus_slicer import create_size_tiers, filter_questions_by_chunks

tiers_path = NQ_DIR / "tiers.json"
if not tiers_path.exists() or True:  # Always regenerate to ensure consistency
    chunks = json.loads((NQ_DIR / "chunks.json").read_text())
    embs = np.load(NQ_DIR / "embeddings.npy")
    total = len(chunks)
    max_tier = min(6000, total)
    tier_sizes = [500, 1000, 2000, 4000, max_tier]
    tier_sizes = sorted(set(t for t in tier_sizes if t <= total))

    log(f"Creating {len(tier_sizes)} size tiers from {total} chunks...")
    tiers = create_size_tiers(chunks, embs, tiers=tier_sizes)
    tiers_data = {str(N): [{"chunk_id": c["chunk_id"]} for c in tier_list]
                  for N, tier_list in tiers.items()}
    tiers_path.write_text(json.dumps(tiers_data))
    log(f"Tiers created: {list(tiers.keys())}")

    # Filter held-out questions per tier
    all_questions = json.loads((NQ_DIR / "eval_heldout.json").read_text())["items"]
    for N_str, tier_ids_dict in tiers_data.items():
        N = int(N_str)
        chunk_ids = {c["chunk_id"] for c in tier_ids_dict}
        filtered = filter_questions_by_chunks(all_questions, chunk_ids)
        eval_path = NQ_DIR / f"eval_tier_{N}.json"
        eval_path.write_text(json.dumps({"items": filtered}))
        log(f"  Tier {N}: {len(filtered)} eval questions")

log("Phase 1 preparation complete!")
print("=" * 60)
print("Next step: Generate FAQs for each tier:")
print("  for N in 500 1000 2000 4000 6000; do")
print("    python -m pipeline.sleep_faq_generator --config examples/natural_questions/config.json \\")
print("      --count $((N * 3)) --output examples/natural_questions/faqs_tier_${N}.json")
print("  done")
print("=" * 60)
