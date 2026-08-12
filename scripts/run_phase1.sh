#!/usr/bin/env bash
# Phase 1 — UC4 Bedrock size sweep on EC2
# Run: cd /home/ubuntu/kvforge && nohup bash scripts/run_phase1.sh > logs/phase1.log 2>&1 &
set -euo pipefail

export GEMINI_API_KEY='AIzaSyAhi-xaDAldgAUa8E7JcBhtzhdBdOg2Nho'
cd /home/ubuntu/kvforge
mkdir -p logs examples/usecase4_bedrock_userguide

VENV="$(pwd)/venv/bin/python3"
CONFIG="examples/usecase4_bedrock_userguide/config.json"
COLLECTION="bedrock-userguide"
DATA_DIR="examples/usecase4_bedrock_userguide"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─── Step 1: Update config for Gemma-4-E2B-it ───────────────────────────
log "Step 1: Updating config for Gemma-4-E2B-it..."
cat > "$CONFIG" << 'JSONEOF'
{
  "use_case_name": "AWS Bedrock User Guide",
  "collection": "bedrock-userguide",
  "version_file": "examples/usecase4_bedrock_userguide/version.json",
  "llm_model": "google/gemma-4-E2B-it",
  "quantization": "4bit",
  "addons": ["indexing", "inference", "training", "background"],
  "addon_config": {
    "indexing": {
      "loader": "jsonl", "embed_model": "BAAI/bge-small-en-v1.5",
      "vector_dim": 384, "vector_store": "qdrant",
      "qdrant_host": "localhost", "qdrant_port": 6333
    },
    "inference": {
      "llm_model": "google/gemma-4-E2B-it",
      "top_k": 5, "max_new_tokens": 128
    },
    "training": {
      "lora_rank": 16, "lora_alpha": 32, "lora_lr": 2e-4,
      "lora_epochs": 1,
      "checkpoint_dir": "examples/usecase4_bedrock_userguide/lora_checkpoints/",
      "replay_db": "examples/usecase4_bedrock_userguide/replay.db",
      "sft_format": "chat"
    },
    "background": {"flush_seconds": 300, "flush_queries": 50}
  },
  "model_library": {
    "google/gemma-4-E2B-it": {"num_layers": 35, "num_kv_heads": 1, "head_dim": 256}
  }
}
JSONEOF

# version.json
echo '{"current_lora_version":0,"checkpoint_path":null,"phase":1,"prs_history":[]}' > "$DATA_DIR/version.json"
log "Config updated."

# ─── Step 2: Check chunks.json exists ─────────────────────────────────
QDRANT_EXTRACT=0; if [ "$QDRANT_EXTRACT" = "1" ] && [ ! -f "$DATA_DIR/chunks.json" ]; then
    log "Step 2: Extracting chunks from Qdrant..."
    $VENV -c "
import json, numpy as np
from pathlib import Path
from vectorstore.registry import get_store

cfg = json.loads(Path('$CONFIG').read_text())
store = get_store(cfg)
chunks, embs = [], []
results, offset = store.scroll('$COLLECTION', limit=1000, with_payload=True, with_vectors=True)
while results:
    for r in results:
        chunks.append({'chunk_id': r.id, 'text': r.payload.get('text','')})
        if hasattr(r, 'vector') and r.vector:
            embs.append(r.vector)
    if offset is None:
        break
    results, offset = store.scroll('$COLLECTION', limit=1000, with_payload=True, with_vectors=True, offset=offset)

np.save('${DATA_DIR}/embeddings.npy', np.array(embs))
Path('${DATA_DIR}/chunks.json').write_text(json.dumps(chunks))
print(f'Extracted {len(chunks)} chunks, {len(embs)} embeddings')
"
else
    log "Already indexed."
fi

# ─── Step 3: Create size tiers ─────────────────────────────────────────
log "Step 3: Creating size tiers..."
$VENV -c "
import json, numpy as np
from pathlib import Path
from tools.corpus_slicer import create_size_tiers, filter_questions_by_chunks

DATA_DIR = Path('$DATA_DIR')
chunks = json.loads((DATA_DIR / 'chunks.json').read_text())
embs = np.load(DATA_DIR / 'embeddings.npy')
total = len(chunks)
tiers = [500, 1000, 2000, 4000, min(6000, total)]
tiers = sorted(set(t for t in tiers if t <= total))

print(f'Creating {len(tiers)} size tiers from {total} chunks...')
tier_data = create_size_tiers(chunks, embs, tiers=tiers)
for N, tier_list in tier_data.items():
    # Save tier chunk IDs
    out_path = DATA_DIR / f'tier_{N}.json'
    out_path.write_text(json.dumps([{'chunk_id': c['chunk_id']} for c in tier_list]))
    print(f'  Tier {N}: {len(tier_list)} chunks')

# Filter eval questions (use existing heldout if available, else empty)
eval_file = DATA_DIR / 'eval_heldout.json'
if eval_file.exists():
    all_q = json.loads(eval_file.read_text())
    items = all_q.get('items', all_q if isinstance(all_q, list) else [])
    source_chunk_ids = {str(c['chunk_id']) for c in chunks}
    filtered = filter_questions_by_chunks(items, source_chunk_ids)
    for N in tiers:
        tier_ids = {c['chunk_id'] for c in json.loads((DATA_DIR/f'tier_{N}.json').read_text())}
        tier_q = filter_questions_by_chunks(items, tier_ids)
        (DATA_DIR / f'eval_tier_{N}.json').write_text(json.dumps({'items': tier_q}))
        print(f'  Tier {N} eval: {len(tier_q)} questions')
    print(f'Total held-out: {len(filtered)} questions')
else:
    print('No eval_heldout.json found — creating placeholder')
    (DATA_DIR / 'eval_heldout.json').write_text(json.dumps({'items': []}))
" 2>&1
log "Tiers created."

# ─── Step 4: Check Gemma model ─────────────────────────────────────────
log "Step 4: Checking Gemma model availability..."
$VENV -c "
from core.model_loader import init, load
init({'llm_model': 'google/gemma-4-E2B-it'})
try:
    model, tokenizer = load()
    print(f'Model loaded: {model.config.model_type}, params={sum(p.numel() for p in model.parameters())/1e6:.0f}M')
    print('MODEL_LOAD_OK')
except Exception as e:
    print(f'Model load failed: {e}')
" 2>&1 | tail -5
log "Model check complete."

# ─── Step 5: Generate FAQs for each tier ───────────────────────────────
log "Step 5: Generating FAQs for each tier..."
for N in 500 1000 2000 4000 6000; do
    FAQ_FILE="$DATA_DIR/faqs_tier_${N}.json"
    if [ -f "$FAQ_FILE" ] && [ $(stat -c%s "$FAQ_FILE" 2>/dev/null || stat -f%z "$FAQ_FILE" 2>/dev/null) -gt 1000 ]; then
        log "  FAQ tier ${N} already exists, skipping"
        continue
    fi
    log "  Generating ${N} FAQs for tier ${N}..."
    $VENV -m pipeline.sleep_faq_generator \
        --config "$CONFIG" \
        --count $((N * 2)) \
        --output "$FAQ_FILE" 2>&1 | tail -3
    log "  FAQ tier ${N} complete"
done

# ─── Step 6: Training + Evaluation for each tier × seed ────────────────
log "Step 6: Starting training + evaluation grid..."
for N in 500 1000 2000 4000 6000; do
    FAQ_FILE="$DATA_DIR/faqs_tier_${N}.json"
    EVAL_FILE="$DATA_DIR/eval_tier_${N}.json"
    TIER_FILE="$DATA_DIR/tier_${N}.json"

    if [ ! -f "$FAQ_FILE" ]; then
        log "  SKIP tier ${N}: no FAQ file"
        continue
    fi

    for SEED in 42 43 44; do
        OUT_DIR="results/absorption/phase1/N${N}_seed${SEED}"
        if [ -d "$OUT_DIR" ] && [ -f "$OUT_DIR/summary.json" ]; then
            log "  SKIP tier ${N} seed ${SEED}: already done"
            continue
        fi

        log "  Training tier ${N} seed ${SEED}..."
        $VENV -m pipeline.lora_trainer \
            --config "$CONFIG" \
            --faqs "$FAQ_FILE" \
            --seed $SEED 2>&1 | tail -3

        log "  Evaluating tier ${N} seed ${SEED}..."
        mkdir -p "$OUT_DIR"
        $VENV -m tools.measure_baseline_fkds \
            --config "$CONFIG" \
            --eval-set "$EVAL_FILE" \
            --output-dir "$OUT_DIR" \
            --modes text_rag parametric \
            --judge-model gemini-2.5-flash 2>&1 | tail -3

        log "  Tier ${N} seed ${SEED} complete"
    done
done

log "=== Phase 1 COMPLETE ==="