#!/usr/bin/env bash
# ============================================================
# compute_kv.sh — Compute and store KV tensors for indexed chunks
#
# Runs Phase 1→2 bridge: reads chunks from the vector store,
# runs them through the LLM to compute KV tensors, and stores
# the serialized tensors back as chunk payloads.
#
# Usage:
#   scripts/compute_kv.sh <config.json>
#
# Required arguments:
#   CONFIG  Path to datasource JSON config file
#
# Prerequisites:
#   - GPU recommended (will work on CPU but very slow)
#   - Documents must already be indexed (run index.sh first)
#
# Example:
#   ./scripts/compute_kv.sh datasource_my-corpus.json
# ============================================================
set -euo pipefail

if ! command -v python &>/dev/null; then
  echo "Error: python not found in PATH" >&2; exit 1
fi

CONFIG="${1:-}"

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "Usage: $0 <config.json>" >&2; exit 1
fi

python -m pipeline.kv_indexer --config "$CONFIG" compute-kv
