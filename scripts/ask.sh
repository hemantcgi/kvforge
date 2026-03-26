#!/usr/bin/env bash
# ============================================================
# ask.sh — Query KVForge from the command line
#
# Usage:
#   scripts/ask.sh <config.json> "<query>"
#
# Required arguments:
#   CONFIG  Path to datasource JSON config file
#   QUERY   The question to ask (quoted string)
#
# Example:
#   ./scripts/ask.sh datasource_my-corpus.json "What is RAG?"
# ============================================================
set -euo pipefail

if ! command -v python &>/dev/null; then
  echo "Error: python not found in PATH" >&2; exit 1
fi

CONFIG="${1:-}"
QUERY="${2:-}"

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "Usage: $0 <config.json> \"<query>\"" >&2; exit 1
fi
if [[ -z "$QUERY" ]]; then
  echo "Usage: $0 <config.json> \"<query>\"" >&2; exit 1
fi

python ask.py --config "$CONFIG" "$QUERY"
