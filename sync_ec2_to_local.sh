#!/usr/bin/env bash
# Pull EC2 outputs into the local working copy.
# Local code remains the source of truth; this script only pulls artifacts/logs.
set -euo pipefail

EC2_USER="${EC2_USER:-ubuntu}"
: "${EC2_HOST:?Set EC2_HOST}"
: "${EC2_PEM:?Set EC2_PEM}"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_REPO="~/kvforge"
SSH_OPTS="-i ${EC2_PEM} -o StrictHostKeyChecking=accept-new"

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
  --exclude='*.pem'
  --exclude='.env'
  --exclude='*.env'
  --exclude='*'
)

rsync -avz --progress \
  "${RSYNC_INCLUDES[@]}" \
  -e "ssh ${SSH_OPTS}" \
  "${EC2_USER}@${EC2_HOST}:${REMOTE_REPO}/" \
  "${LOCAL_REPO}/"
