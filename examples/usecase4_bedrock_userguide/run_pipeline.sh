#!/usr/bin/env bash
# run_pipeline.sh — Full KVForge pipeline for Use Case 4: Amazon Bedrock User Guide + Qdrant
#
# Prerequisites
# -------------
#   - Docker running with Qdrant:
#       docker run -d -p 6333:6333 qdrant/qdrant
#   - Python venv activated with KVForge dependencies
#   - HuggingFace token set (for Llama access):
#       export HF_TOKEN=hf_...
#   - GPU recommended for KV computation and LoRA training
#
# Run from repo root:
#   bash examples/usecase4_bedrock_userguide/run_pipeline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UC_DIR="$REPO_ROOT/examples/usecase4_bedrock_userguide"
CONFIG="$UC_DIR/config.json"
FAQS="$UC_DIR/faqs.json"
PDF="$UC_DIR/data/amazon-bedrock-user-guide.pdf"

cd "$REPO_ROOT"

echo "================================================================"
echo " KVForge — Use Case 4: Amazon Bedrock User Guide (Qdrant)"
echo "================================================================"

# ── Step 0: Verify PDF is present ────────────────────────────────────────────
echo ""
echo "[ Step 0 ] Verifying source PDF …"
if [ ! -f "$PDF" ]; then
    echo "  ERROR: PDF not found at $PDF" >&2
    echo "  Expected: examples/usecase4_bedrock_userguide/data/amazon-bedrock-user-guide.pdf" >&2
    exit 1
fi
echo "  Found: $PDF"

# ── Step 1: Index the PDF into Qdrant ────────────────────────────────────────
echo ""
echo "[ Step 1 ] Indexing Amazon Bedrock User Guide into Qdrant …"
python kvforge.py index --config "$CONFIG" --source "$PDF"

# ── Step 2: Compute KV tensors for all chunks ─────────────────────────────────
echo ""
echo "[ Step 2 ] Computing KV tensors (requires GPU) …"
python -m pipeline.kv_indexer --config "$CONFIG" compute-kv

# ── Step 3: Use pre-generated FAQs or generate new ones ──────────────────────
echo ""
echo "[ Step 3 ] Preparing FAQs for PRS evaluation …"
if [ ! -f "$FAQS" ]; then
    echo "  Generating FAQs from corpus …"
    python tools/generate_faqs.py \
        --config "$CONFIG" \
        --count 50 \
        --output "$FAQS"
else
    echo "  Using pre-generated FAQs at $FAQS"
fi

# ── Step 4: Fine-tune LoRA on Q&A pairs ──────────────────────────────────────
echo ""
echo "[ Step 4 ] Fine-tuning LoRA (requires GPU) …"
python -m pipeline.lora_trainer \
    --config "$CONFIG" \
    --faqs "$FAQS"

# ── Step 5: Recompute KV with updated model weights ───────────────────────────
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
python ask.py --config "$CONFIG" "What is Amazon Bedrock?"
echo "---"
python ask.py --config "$CONFIG" "Which foundation models are available in Bedrock?"
echo "---"
python ask.py --config "$CONFIG" "How do I fine-tune a model in Amazon Bedrock?"
echo "---"
python ask.py --config "$CONFIG" "What are Bedrock Agents and how do they work?"

echo ""
echo "================================================================"
echo " Pipeline complete!"
echo " Start the monitoring dashboard with:"
echo "   python -m pipeline.monitoring_dashboard --config $CONFIG"
echo " Open http://localhost:8084"
echo "================================================================"
