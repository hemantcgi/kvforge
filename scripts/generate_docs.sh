#!/usr/bin/env bash
# ============================================================
# generate_docs.sh — Auto-generate API reference from docstrings
#
# Uses pdoc (v14+) to generate HTML documentation from all Python
# module docstrings. Output is written to docs/api/generated/
# and is NOT committed to git (see .gitignore).
#
# Usage:
#   scripts/generate_docs.sh [output_dir]
#
# Optional arguments:
#   OUTPUT_DIR  Where to write generated HTML (default: docs/api/generated)
#
# Prerequisites:
#   pip install pdoc
#
# Example:
#   ./scripts/generate_docs.sh
#   ./scripts/generate_docs.sh /tmp/smartqdrant-docs
# ============================================================
set -euo pipefail

# NOTE: This script intentionally omits the CONFIG validation block used by
# other scripts — pdoc operates on source modules directly and requires no
# datasource config file.

if ! command -v python &>/dev/null; then
  echo "Error: python not found in PATH" >&2; exit 1
fi

if ! python -c "import pdoc" 2>/dev/null; then
  echo "Error: pdoc not installed. Run: pip install pdoc" >&2; exit 1
fi

OUTPUT_DIR="${1:-docs/api/generated}"
mkdir -p "$OUTPUT_DIR"

python -m pdoc -o "$OUTPUT_DIR" \
  core \
  pipeline embeddings ingestion vectorstore tools

echo "Documentation written to: $OUTPUT_DIR"
