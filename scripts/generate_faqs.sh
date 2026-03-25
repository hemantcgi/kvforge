#!/usr/bin/env bash
# ============================================================
# generate_faqs.sh — Auto-generate FAQ pairs from indexed corpus
#
# Samples chunks from the vector store, calls the LLM to produce
# Q/A pairs in "Q: ... A: ..." format, and saves them to a JSON file
# suitable for train_lora.sh and evaluate_prs.sh.
#
# Usage:
#   scripts/generate_faqs.sh <config.json> <output.json> [count]
#
# Required arguments:
#   CONFIG  Path to datasource JSON config file
#   OUTPUT  Output JSON file path for generated FAQs
#
# Optional arguments:
#   COUNT   Number of FAQ pairs to generate (default: 50)
#
# Prerequisites:
#   - Documents must be indexed first (run index.sh)
#   - GPU recommended for LLM calls
#
# Example:
#   ./scripts/generate_faqs.sh datasource_my-corpus.json my-corpus_faqs.json
#   ./scripts/generate_faqs.sh datasource_my-corpus.json my-corpus_faqs.json 100
# ============================================================
set -euo pipefail

if ! command -v python &>/dev/null; then
  echo "Error: python not found in PATH" >&2; exit 1
fi

CONFIG="${1:-}"
OUTPUT="${2:-}"
COUNT="${3:-50}"

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "Usage: $0 <config.json> <output.json> [count]" >&2; exit 1
fi
if [[ -z "$OUTPUT" ]]; then
  echo "Usage: $0 <config.json> <output.json> [count]" >&2; exit 1
fi

python tools/generate_faqs.py --config "$CONFIG" --output "$OUTPUT" --count "$COUNT"
