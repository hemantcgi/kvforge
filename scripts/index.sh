#!/usr/bin/env bash
# ============================================================
# index.sh — Load, embed, and index documents into the vector store
#
# Usage:
#   scripts/index.sh <config.json> <source_path>
#
# Required arguments:
#   CONFIG  Path to datasource JSON config file
#   SOURCE  Path to source documents (file or directory)
#
# Example:
#   ./scripts/index.sh datasource_my-corpus.json ./docs/
#   ./scripts/index.sh datasource_my-corpus.json my_document.pdf
# ============================================================
set -euo pipefail

if ! command -v python &>/dev/null; then
  echo "Error: python not found in PATH" >&2; exit 1
fi

CONFIG="${1:-}"
SOURCE="${2:-}"

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "Usage: $0 <config.json> <source_path>" >&2; exit 1
fi
if [[ -z "$SOURCE" ]]; then
  echo "Usage: $0 <config.json> <source_path>" >&2; exit 1
fi

python kvforge.py index --config "$CONFIG" --source "$SOURCE"
