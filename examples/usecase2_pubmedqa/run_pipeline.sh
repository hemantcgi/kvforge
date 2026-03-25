#!/usr/bin/env bash
# run_pipeline.sh — Full SmartQdrant pipeline for Use Case 2: PubMedQA + ChromaDB
#
# Prerequisites
# -------------
#   - No Docker needed — ChromaDB runs in-process
#   - Python venv activated with SmartQdrant dependencies
#   - HuggingFace token set (for Llama access):
#       export HF_TOKEN=hf_...
#   - GPU recommended for KV computation and LoRA training
#   - Install extras: pip install datasets chromadb
#
# Run from repo root:
#   bash examples/usecase2_pubmedqa/run_pipeline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UC_DIR="$REPO_ROOT/examples/usecase2_pubmedqa"
CONFIG="$UC_DIR/config.json"
FAQS="$UC_DIR/faqs.json"
CORPUS="$UC_DIR/data/corpus.jsonl"

cd "$REPO_ROOT"

echo "================================================================"
echo " SmartQdrant — Use Case 2: Biomedical Q&A (ChromaDB)"
echo "================================================================"

# ── Step 0: Download and prepare data ────────────────────────────────────────
echo ""
echo "[ Step 0 ] Preparing PubMedQA dataset …"
if [ ! -f "$CORPUS" ]; then
    python examples/usecase2_pubmedqa/setup.py
else
    echo "  Data already prepared, skipping download."
fi

# ── Step 1: Index corpus into ChromaDB ───────────────────────────────────────
echo ""
echo "[ Step 1 ] Indexing corpus into ChromaDB …"
python smartqdrant.py index --config "$CONFIG" --source "$CORPUS"

# ── Step 2: Compute KV tensors ────────────────────────────────────────────────
echo ""
echo "[ Step 2 ] Computing KV tensors (requires GPU) …"
python -m pipeline.kv_indexer --config "$CONFIG" compute-kv

# ── Step 3: Use pre-generated FAQs ────────────────────────────────────────────
echo ""
echo "[ Step 3 ] Using pre-generated PubMedQA FAQs …"
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
echo "[ Step 4 ] Fine-tuning LoRA on biomedical Q&A (requires GPU) …"
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
echo "[ Step 7 ] Running biomedical example queries …"
echo "---"

python ask.py --config "$CONFIG" \
    "Does metformin reduce cardiovascular risk in type 2 diabetes patients?"
echo "---"
python ask.py --config "$CONFIG" \
    "What is the effect of aspirin on colorectal cancer prevention?"
echo "---"
python ask.py --config "$CONFIG" \
    "Is there a link between gut microbiome and depression?"

echo ""
echo "================================================================"
echo " Pipeline complete!"
echo " ChromaDB data stored at: examples/usecase2_pubmedqa/.chroma/"
echo " Start the monitoring dashboard with:"
echo "   python -m pipeline.monitoring_dashboard --config $CONFIG"
echo "================================================================"
