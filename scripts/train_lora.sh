#!/usr/bin/env bash
# ============================================================
# train_lora.sh — Fine-tune the LLM with LoRA on FAQ pairs
#
# Uses PEFT LoRA to fine-tune the configured LLM on question/answer
# pairs from a FAQs JSON file. Saves adapter weights to
# the checkpoint_dir specified in the config.
#
# Usage:
#   scripts/train_lora.sh <config.json> <faqs.json>
#
# Required arguments:
#   CONFIG  Path to datasource JSON config file
#   FAQS    Path to FAQs JSON file (array of {question, answer} objects)
#
# Prerequisites:
#   - GPU required (LoRA fine-tuning is GPU-only)
#
# Example:
#   ./scripts/train_lora.sh datasource_my-corpus.json my-corpus_faqs.json
# ============================================================
set -euo pipefail

if ! command -v python &>/dev/null; then
  echo "Error: python not found in PATH" >&2; exit 1
fi

CONFIG="${1:-}"
FAQS="${2:-}"

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "Usage: $0 <config.json> <faqs.json>" >&2; exit 1
fi
if [[ -z "$FAQS" || ! -f "$FAQS" ]]; then
  echo "Error: FAQS file not found: $FAQS" >&2
  echo "Usage: $0 <config.json> <faqs.json>" >&2; exit 1
fi

python -m pipeline.lora_trainer --config "$CONFIG" --faqs "$FAQS"
