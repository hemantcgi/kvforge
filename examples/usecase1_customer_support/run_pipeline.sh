#!/usr/bin/env bash
# run_pipeline.sh — Full SmartQdrant pipeline for Use Case 1: Customer Support + Qdrant
#
# Prerequisites
# -------------
#   - Docker running with Qdrant:
#       docker run -d -p 6333:6333 qdrant/qdrant
#   - Python venv activated with SmartQdrant dependencies
#   - HuggingFace token set (for Llama access):
#       export HF_TOKEN=hf_...
#   - GPU recommended for KV computation and LoRA training
#
# Run from repo root:
#   bash examples/usecase1_customer_support/run_pipeline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UC_DIR="$REPO_ROOT/examples/usecase1_customer_support"
CONFIG="$UC_DIR/config.json"
FAQS="$UC_DIR/faqs.json"
CORPUS="$UC_DIR/data/corpus.jsonl"

cd "$REPO_ROOT"

echo "================================================================"
echo " SmartQdrant — Use Case 1: Customer Support Q&A (Qdrant)"
echo "================================================================"

# ── Step 0: Download and prepare data ────────────────────────────────────────
echo ""
echo "[ Step 0 ] Preparing dataset …"
if [ ! -f "$CORPUS" ]; then
    python examples/usecase1_customer_support/setup.py
else
    echo "  Data already prepared, skipping download."
fi

# ── Step 1: Index corpus into Qdrant ─────────────────────────────────────────
echo ""
echo "[ Step 1 ] Indexing corpus into Qdrant …"
python smartqdrant.py index --config "$CONFIG" --source "$CORPUS"

# ── Step 2: Compute KV tensors for all chunks ─────────────────────────────────
echo ""
echo "[ Step 2 ] Computing KV tensors (requires GPU) …"
python kv_indexer.py --config "$CONFIG" compute-kv

# ── Step 3: Generate FAQs (or use pre-generated) ─────────────────────────────
echo ""
echo "[ Step 3 ] Generating FAQs for PRS evaluation …"
if [ ! -f "$FAQS" ]; then
    python tools/generate_faqs.py \
        --config "$CONFIG" \
        --count 50 \
        --output "$FAQS"
else
    echo "  FAQs already exist at $FAQS, skipping generation."
fi

# ── Step 4: Fine-tune LoRA on Q&A pairs ──────────────────────────────────────
echo ""
echo "[ Step 4 ] Fine-tuning LoRA (requires GPU) …"
python lora_trainer.py \
    --config "$CONFIG" \
    --faqs "$FAQS"

# ── Step 5: Recompute KV with updated model weights ───────────────────────────
echo ""
echo "[ Step 5 ] Recomputing KV tensors with updated LoRA weights …"
python kv_indexer.py --config "$CONFIG" compute-kv

# ── Step 6: Evaluate PRS ──────────────────────────────────────────────────────
echo ""
echo "[ Step 6 ] Evaluating Parametric Readiness Score (PRS) …"
python prs_evaluator.py \
    --config "$CONFIG" \
    --faqs "$FAQS" \
    --sample 30

# ── Step 7: Example queries ───────────────────────────────────────────────────
echo ""
echo "[ Step 7 ] Running example queries …"
echo "---"

python ask.py --config "$CONFIG" "How do I cancel my subscription?"
echo "---"
python ask.py --config "$CONFIG" "What is your refund policy?"
echo "---"
python ask.py --config "$CONFIG" "How do I reset my password?"

echo ""
echo "================================================================"
echo " Pipeline complete!"
echo " Start the monitoring dashboard with:"
echo "   python monitoring_dashboard.py --config $CONFIG"
echo "================================================================"
