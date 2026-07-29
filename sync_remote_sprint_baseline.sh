#!/usr/bin/env bash
# Fast-forward the local sprint/0-baseline branch from smartqdrant.
set -euo pipefail

REPO="/Users/hemant/Downloads/RoPE/KVForge"
LOG="$REPO/logs/git_pull_sprint_baseline.log"

mkdir -p "$REPO/logs"
{
  echo "$(date): pulling smartqdrant sprint/0-baseline..."
  cd "$REPO"
  git checkout sprint/0-baseline
  git pull --ff-only smartqdrant sprint/0-baseline
  echo "$(date): done"
} >> "$LOG" 2>&1
