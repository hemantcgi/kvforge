#!/usr/bin/env bash
# Pull EC2 outputs into the local working copy.
# Local code remains the source of truth; this script only pulls artifacts/logs.
set -euo pipefail

EC2_USER="${EC2_USER:-ubuntu}"
EC2_HOST="${EC2_HOST:-13.217.195.243}"
EC2_PEM="${EC2_PEM:-/Users/hemant/Downloads/RoPE/g5.x.pem}"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_REPO="~/kvforge"
SSH_OPTS="-i ${EC2_PEM} -o StrictHostKeyChecking=no"

# Include the artifacts we want from EC2.
RSYNC_INCLUDES=(
  --include='logs/'
  --include='logs/**'
  --include='examples/'
  --include='examples/usecase4_bedrock_userguide/'
  --include='examples/usecase4_bedrock_userguide/*.json'
  --include='examples/usecase4_bedrock_userguide/*.log'
  --include='examples/usecase4_bedrock_userguide/lora_checkpoints/'
  --include='examples/usecase4_bedrock_userguide/lora_checkpoints/**'
  --include='examples/usecase4_bedrock_userguide/*.db'
  --exclude='*'
)

rsync -avz --progress \
  "${RSYNC_INCLUDES[@]}" \
  -e "ssh ${SSH_OPTS}" \
  "${EC2_USER}@${EC2_HOST}:${REMOTE_REPO}/" \
  "${LOCAL_REPO}/"
