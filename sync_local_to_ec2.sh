#!/usr/bin/env bash
# Push local source/config files to EC2 so the running code is always in sync.
set -euo pipefail

EC2_USER="${EC2_USER:-ubuntu}"
: "${EC2_HOST:?Set EC2_HOST}"
: "${EC2_PEM:?Set EC2_PEM}"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_REPO="~/kvforge"

RSYNC_EXCLUDES=(
  --exclude='venv/'
  --exclude='venv_gpu/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.faiss/'
  --exclude='.chroma/'
  --exclude='logs/'
  --exclude='lora_checkpoints/'
  --exclude='.git/'
  --exclude='.GCC/'
  --exclude='.env'
  --exclude='*.env'
  --exclude='*.pem'
  --exclude='*.safetensors'
  --exclude='*.bin'
  --exclude='*.db'
  --exclude='*.log'
  --exclude='*.png'
  --exclude='*.jpg'
  --exclude='*.jpeg'
  --exclude='query_pool_*.json'
  --exclude='teacher_pairs_*.json'
  --exclude='on_policy_samples_*.json'
  --exclude='distill_pairs_*.json'
  --exclude='eval_*.json'
)

rsync -avz "${RSYNC_EXCLUDES[@]}" \
  -e "ssh -i ${EC2_PEM} -o StrictHostKeyChecking=accept-new" \
  "${LOCAL_REPO}/" \
  "${EC2_USER}@${EC2_HOST}:${REMOTE_REPO}/"
