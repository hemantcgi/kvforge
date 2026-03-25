#!/usr/bin/env bash
# ============================================================
# dashboard.sh — Start the SmartQdrant monitoring dashboard
#
# Usage:
#   scripts/dashboard.sh <config.json> [port]
#
# Required arguments:
#   CONFIG  Path to datasource JSON config file
#
# Optional arguments:
#   PORT    Port to listen on (default: 8080)
#
# Example:
#   ./scripts/dashboard.sh datasource_my-corpus.json
#   ./scripts/dashboard.sh datasource_my-corpus.json 9090
# ============================================================
set -euo pipefail

if ! command -v python &>/dev/null; then
  echo "Error: python not found in PATH" >&2; exit 1
fi

CONFIG="${1:-}"
PORT="${2:-8080}"

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "Usage: $0 <config.json> [port]" >&2; exit 1
fi

python -m pipeline.monitoring_dashboard --config "$CONFIG" --port "$PORT"
