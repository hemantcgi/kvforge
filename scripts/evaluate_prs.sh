#!/usr/bin/env bash
# ============================================================
# evaluate_prs.sh — Evaluate Parametric Readiness Score (PRS)
#
# Runs the PRS evaluation pipeline: asks the configured LLM each FAQ
# question and scores accuracy, calibration, and consistency.
# PRS = 0.5*accuracy + 0.3*calibration + 0.2*consistency.
# If PRS >= prs_threshold in config, the system advances to Phase 3.
#
# Usage:
#   scripts/evaluate_prs.sh <config.json> <faqs.json>
#
# Required arguments:
#   CONFIG  Path to datasource JSON config file
#   FAQS    Path to FAQs JSON file (array of {question, answer} objects)
#
# Example:
#   ./scripts/evaluate_prs.sh datasource_my-corpus.json my-corpus_faqs.json
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

python -m pipeline.prs_evaluator --config "$CONFIG" --faqs "$FAQS"
