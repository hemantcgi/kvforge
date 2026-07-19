#!/usr/bin/env bash
# Push local source/config files to EC2 so the running code is always in sync.
set -euo pipefail

EC2_USER="${EC2_USER:-ubuntu}"
EC2_HOST="${EC2_HOST:-13.217.195.243}"
EC2_PEM="${EC2_PEM:-/Users/hemant/Downloads/RoPE/g5.x.pem}"
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

rsync -avz --update "${RSYNC_EXCLUDES[@]}" \
  -e "ssh -i ${EC2_PEM} -o StrictHostKeyChecking=no" \
  "${LOCAL_REPO}/" \
  "${EC2_USER}@${EC2_HOST}:${REMOTE_REPO}/"
