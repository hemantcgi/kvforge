#!/usr/bin/env bash
# run_pipeline.sh — Full SmartQdrant pipeline for Use Case 3: SQuAD v2 + FAISS
#
# Prerequisites
# -------------
#   - No Docker, no external services — FAISS runs fully in-process
#   - Python venv activated with SmartQdrant dependencies
#   - HuggingFace token set (for Llama access):
#       export HF_TOKEN=hf_...
#   - GPU recommended for KV computation and LoRA training
#   - Install extras: pip install datasets faiss-cpu
#
# Run from repo root:
#   bash examples/usecase3_squad/run_pipeline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UC_DIR="$REPO_ROOT/examples/usecase3_squad"
CONFIG="$UC_DIR/config.json"
FAQS="$UC_DIR/faqs.json"
CORPUS="$UC_DIR/data/corpus.jsonl"

cd "$REPO_ROOT"

echo "================================================================"
echo " SmartQdrant — Use Case 3: General Knowledge Q&A (FAISS)"
echo "================================================================"

# ── Step 0: Download and prepare data ────────────────────────────────────────
echo ""
echo "[ Step 0 ] Preparing SQuAD v2 dataset …"
if [ ! -f "$CORPUS" ]; then
    python examples/usecase3_squad/setup.py
else
    echo "  Data already prepared, skipping download."
fi

# ── Step 1: Index corpus into FAISS ──────────────────────────────────────────
echo ""
echo "[ Step 1 ] Indexing corpus into FAISS (in-process, no server) …"
python smartqdrant.py index --config "$CONFIG" --source "$CORPUS"

# ── Step 2: Compute KV tensors ────────────────────────────────────────────────
echo ""
echo "[ Step 2 ] Computing KV tensors (requires GPU) …"
python -m pipeline.kv_indexer --config "$CONFIG" compute-kv

# ── Step 3: Use pre-generated FAQs ────────────────────────────────────────────
echo ""
echo "[ Step 3 ] Using SQuAD v2 answerable Q&A pairs as FAQs …"
if [ ! -f "$FAQS" ]; then
    python tools/generate_faqs.py \
        --config "$CONFIG" \
        --count 50 \
        --output "$FAQS"
else
    echo "  FAQs present at $FAQS"
fi

# ── Step 4: Fine-tune LoRA ────────────────────────────────────────────────────
echo ""
echo "[ Step 4 ] Fine-tuning LoRA on Wikipedia Q&A (requires GPU) …"
python -m pipeline.lora_trainer \
    --config "$CONFIG" \
    --faqs "$FAQS"

# ── Step 5: Recompute KV with updated weights ─────────────────────────────────
echo ""
echo "[ Step 5 ] Recomputing KV tensors with updated LoRA weights …"
python -m pipeline.kv_indexer --config "$CONFIG" compute-kv

# ── Step 6: Evaluate PRS ──────────────────────────────────────────────────────
echo ""
echo "[ Step 6 ] Evaluating Parametric Readiness Score (PRS) …"
python -m pipeline.prs_evaluator \
    --config "$CONFIG" \
    --faqs "$FAQS" \
    --sample 30

# ── Step 7: Example queries ───────────────────────────────────────────────────
echo ""
echo "[ Step 7 ] Running example queries …"
echo "---"

python ask.py --config "$CONFIG" \
    "In what year did the French Revolution begin?"
echo "---"
python ask.py --config "$CONFIG" \
    "Who wrote the theory of relativity?"
echo "---"
python ask.py --config "$CONFIG" \
    "What is the capital of Australia?"

echo ""
echo "================================================================"
echo " Pipeline complete!"
echo " FAISS index stored at: examples/usecase3_squad/.faiss/"
echo " Start the monitoring dashboard with:"
echo "   python -m pipeline.monitoring_dashboard --config $CONFIG"
echo "================================================================"
