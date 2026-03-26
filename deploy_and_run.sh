#!/usr/bin/env bash
set -euo pipefail

# ─── Config ───────────────────────────────────────────────────────────────────
EC2_USER="ubuntu"
EC2_HOST="13.221.47.200"
EC2_PEM="/Users/hemant/Downloads/RoPE/g5.x.pem"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_REPO="~/kvforge"
VENV="~/qdrant/venv"
SSH="ssh -i $EC2_PEM -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST"

# ─── Helper ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─── Step 1/7 — rsync repo to EC2 ────────────────────────────────────────────
log "Step 1/7 — Syncing repo to EC2 ($EC2_USER@$EC2_HOST:$REMOTE_REPO) ..."

RSYNC_EXCLUDES=(
  --exclude='venv/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.faiss/'
  --exclude='.chroma/'
  --exclude='logs/'
  --exclude='lora_checkpoints/'
  --exclude='.git/'
)
rsync -avz --progress "${RSYNC_EXCLUDES[@]}" \
  -e "ssh -i $EC2_PEM -o StrictHostKeyChecking=no" \
  "$LOCAL_REPO/" \
  "$EC2_USER@$EC2_HOST:$REMOTE_REPO/"

log "Step 1/7 — Sync complete."

# ─── Step 2/7 — Install Python deps ──────────────────────────────────────────
log "Step 2/7 — Installing Python dependencies into existing venv (quiet) ..."

$SSH "bash -c 'source ~/.bashrc && $VENV/bin/pip install -r $REMOTE_REPO/requirements_gpu.txt -q'"

log "Step 2/7 — Dependencies installed."

# ─── Step 3/7 — Start Qdrant via Docker ──────────────────────────────────────
log "Step 3/7 — Starting Qdrant via Docker (idempotent + persistent volume) ..."

$SSH "bash -c 'docker start qdrant 2>/dev/null || docker run -d --name qdrant -p 6333:6333 -v ~/qdrant_storage:/qdrant/storage qdrant/qdrant'"

log "Step 3/7 — Qdrant running."

# ─── Step 4/7 — Migrate UC4 weights + write version.json ─────────────────────
log "Step 4/7 — Migrating UC4 LoRA v3 weights and writing version.json ..."

$SSH "bash -c '
  set -euo pipefail
  DEST=$REMOTE_REPO/examples/usecase4_bedrock_userguide/lora_checkpoints/v3
  mkdir -p \"\$DEST\"
  cp -r ~/qdrant/lora_checkpoints/bedrock/v3/* \"\$DEST/\"
  cat > $REMOTE_REPO/examples/usecase4_bedrock_userguide/version.json << EOF
{
  \"current_lora_version\": 3,
  \"checkpoint_path\": \"examples/usecase4_bedrock_userguide/lora_checkpoints/v3/\",
  \"phase\": 3,
  \"prs_history\": [],
  \"known_good_queries\": []
}
EOF
  echo \"UC4 weights migrated to \$DEST\"
'"

log "Step 4/7 — UC4 weights migrated."

# ─── Step 5/7 — Staggered launch ─────────────────────────────────────────────
log "Step 5/7 — Staggered launch of use-case workers ..."

# Ensure logs directory exists on EC2
$SSH "mkdir -p $REMOTE_REPO/logs"

# T+0s — UC4 on GPU 3 (fast-path)
log "  T+0s   — Launching UC4 on GPU 3 ..."
$SSH "bash -c '
  source ~/.bashrc
  cd $REMOTE_REPO
  nohup bash -c \"source ~/.bashrc && cd $REMOTE_REPO && CUDA_VISIBLE_DEVICES=3 bash examples/usecase4_bedrock_userguide/start_uc4_dashboard.sh\" \
    >> $REMOTE_REPO/logs/uc4.log 2>&1 &
  echo \"UC4 worker PID: \$!\"
'"

log "  Waiting 60s before UC1 ..."
sleep 60

# T+60s — UC1 on GPU 0 (full pipeline)
log "  T+60s  — Launching UC1 on GPU 0 ..."
$SSH "bash -c '
  nohup bash -c \"source ~/.bashrc && cd $REMOTE_REPO && CUDA_VISIBLE_DEVICES=0 bash examples/usecase1_customer_support/run_pipeline.sh\" \
    >> $REMOTE_REPO/logs/uc1.log 2>&1 &
  echo \"UC1 worker PID: \$!\"
'"

log "  Waiting 120s before UC2 ..."
sleep 120

# T+180s — UC2 on GPU 1 (full pipeline)
log "  T+180s — Launching UC2 on GPU 1 ..."
$SSH "bash -c '
  nohup bash -c \"source ~/.bashrc && cd $REMOTE_REPO && CUDA_VISIBLE_DEVICES=1 bash examples/usecase2_pubmedqa/run_pipeline.sh\" \
    >> $REMOTE_REPO/logs/uc2.log 2>&1 &
  echo \"UC2 worker PID: \$!\"
'"

log "  Waiting 120s before UC3 ..."
sleep 120

# T+300s — UC3 on GPU 2 (full pipeline)
log "  T+300s — Launching UC3 on GPU 2 ..."
$SSH "bash -c '
  nohup bash -c \"source ~/.bashrc && cd $REMOTE_REPO && CUDA_VISIBLE_DEVICES=2 bash examples/usecase3_squad/run_pipeline.sh\" \
    >> $REMOTE_REPO/logs/uc3.log 2>&1 &
  echo \"UC3 worker PID: \$!\"
'"

log "Step 5/7 — All workers launched."

# ─── Step 6/7 — Start dashboards and portal ──────────────────────────────────
log "Step 6/7 — Starting dashboards and KVForge portal ..."

$SSH "bash -c '
  source ~/.bashrc
  cd $REMOTE_REPO

  fuser -k 8080/tcp 2>/dev/null || true
  for port in 8081 8082 8083 8084; do
    fuser -k \${port}/tcp 2>/dev/null || true
  done
  sleep 2

  nohup bash -c \"source ~/.bashrc && cd $REMOTE_REPO && bash examples/usecase1_customer_support/start_dashboard.sh\" \
    >> $REMOTE_REPO/logs/dashboard_uc1.log 2>&1 &

  nohup bash -c \"source ~/.bashrc && cd $REMOTE_REPO && bash examples/usecase2_pubmedqa/start_dashboard.sh\" \
    >> $REMOTE_REPO/logs/dashboard_uc2.log 2>&1 &

  nohup bash -c \"source ~/.bashrc && cd $REMOTE_REPO && bash examples/usecase3_squad/start_dashboard.sh\" \
    >> $REMOTE_REPO/logs/dashboard_uc3.log 2>&1 &

  # UC4 dashboard already started in Step 5 via start_uc4_dashboard.sh

  nohup bash -c \"source ~/.bashrc && cd $REMOTE_REPO && $VENV/bin/python3 kvforge_portal.py --port 8080\" \
    >> $REMOTE_REPO/logs/portal.log 2>&1 &

  echo \"All dashboards and portal started.\"
'"

log "Step 6/7 — Dashboards and portal started."

# ─── Step 7/7 — Print access URLs ────────────────────────────────────────────
log "Step 7/7 — Deployment complete."

cat <<'URLS'

════════════════════════════════════════════════════════
 KVForge Multi-GPU Deployment — Access URLs
════════════════════════════════════════════════════════
 KVForge portal  : http://13.221.47.200:8080
 UC1 dashboard   : http://13.221.47.200:8081  (Customer Support)
 UC2 dashboard   : http://13.221.47.200:8082  (PubMedQA)
 UC3 dashboard   : http://13.221.47.200:8083  (SQuAD)
 UC4 dashboard   : http://13.221.47.200:8084  (Bedrock User Guide)
════════════════════════════════════════════════════════

 Logs on EC2 at ~/kvforge/logs/:
   uc1.log, uc2.log, uc3.log, uc4.log
   dashboard_uc{1,2,3}.log, portal.log

 Tail a log:  ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@13.221.47.200 'tail -f ~/kvforge/logs/uc1.log'

URLS
