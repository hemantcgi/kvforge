#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh — Full SmartQdrant Phase 1→2→3 pipeline
#
# Runs the complete pipeline for a new corpus:
#   1. Index documents (embed + upsert to vector store)
#   2. Generate FAQ pairs from the indexed corpus
#   3. Compute KV tensors (Phase 1→2 bridge)
#   4. Fine-tune with LoRA (Phase 2)
#   5. Recompute KV with updated weights
#   6. Evaluate PRS (gate for Phase 3)
#
# Usage:
#   scripts/run_pipeline.sh <config.json> <source_path> <faqs.json>
#
# Required arguments:
#   CONFIG  Path to datasource JSON config file
#   SOURCE  Path to source documents (file or directory)
#   FAQS    Path to FAQs JSON file (will be created by generate_faqs if absent)
#
# Prerequisites:
#   - GPU recommended (KV computation and LoRA training require GPU)
#   - Qdrant Docker running (if vector_store=qdrant in config)
#   - HuggingFace token set (if using a gated model like Llama 3)
#
# Example:
#   ./scripts/run_pipeline.sh datasource_my-corpus.json ./docs/ my-corpus_faqs.json
# ============================================================
set -euo pipefail

if ! command -v python &>/dev/null; then
  echo "Error: python not found in PATH" >&2; exit 1
fi

CONFIG="${1:-}"
SOURCE="${2:-}"
FAQS="${3:-}"

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "Usage: $0 <config.json> <source_path> <faqs.json>" >&2; exit 1
fi
if [[ -z "$SOURCE" ]]; then
  echo "Usage: $0 <config.json> <source_path> <faqs.json>" >&2; exit 1
fi
if [[ -z "$FAQS" ]]; then
  echo "Usage: $0 <config.json> <source_path> <faqs.json>" >&2; exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "SmartQdrant Full Pipeline"
echo "Config:  $CONFIG"
echo "Source:  $SOURCE"
echo "FAQs:    $FAQS"
echo "============================================================"

# Step 1: Index documents
printf "\n[1/6] Indexing documents...\n"
python smartqdrant.py index --config "$CONFIG" --source "$SOURCE"

# Step 2: Generate FAQs (if file doesn't exist yet)
if [[ ! -f "$FAQS" ]]; then
  printf "\n[2/6] Generating FAQ pairs...\n"
  python tools/generate_faqs.py --config "$CONFIG" --output "$FAQS" --count 50
else
  printf "\n[2/6] FAQs file already exists, skipping generation: %s\n" "$FAQS"
fi

# Step 3: Compute KV tensors
printf "\n[3/6] Computing KV tensors (Phase 1->2)...\n"
python -m pipeline.kv_indexer --config "$CONFIG" compute-kv

# Step 4: LoRA fine-tuning
printf "\n[4/6] LoRA fine-tuning...\n"
python -m pipeline.lora_trainer --config "$CONFIG" --faqs "$FAQS"

# Step 5: Recompute KV with updated weights
printf "\n[5/6] Recomputing KV tensors with updated weights...\n"
python -m pipeline.kv_indexer --config "$CONFIG" compute-kv

# Step 6: PRS evaluation
printf "\n[6/6] Evaluating Parametric Readiness Score...\n"
python -m pipeline.prs_evaluator --config "$CONFIG" --faqs "$FAQS"

printf "\n============================================================\n"
echo "Pipeline complete. Check PRS output above."
echo "If PRS >= prs_threshold, the system has advanced to Phase 3."
echo "============================================================"
