#!/bin/bash
# Step-5 orchestrator: waits for the R2 RAFT training to finish and the adapter to
# save, then runs the 4-arm RAG-mode prediction dump SEQUENTIALLY (single A10G).
# Hard guard: only dumps if the final adapter exists — a crash/DEAD never triggers
# a garbage dump. Launch detached: setsid nohup ... </dev/null &
set -u
cd ~/kvforge
KROOT=~/kvforge
CKPT="examples/usecase4_bedrock_userguide/lora_checkpoints/raft_R2_allLinear"
RAFT="$KROOT/$CKPT"
PROMO="$KROOT/examples/usecase4_bedrock_userguide/lora_checkpoints/promo_E2B_allLinear_60kqa_rw"
OUT="examples/usecase4_bedrock_userguide"
LOG="$KROOT/phase2_dump.log"
PY="$KROOT/venv_gpu/bin/python"
export PYTHONPATH="$KROOT" PYTHONUNBUFFERED=1
ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[orch $(ts)] armed; waiting for trainer to exit + adapter save" >> "$LOG"
# Wait on the REAL python trainer (comm=python) only — NOT the launcher-bash
# wrapper, whose cmdline also contains "pipeline.lora_trainer" and lingers.
trainer_alive() { ps -eo comm,args | awk '$1=="python" && /pipeline\.lora_trainer/ {f=1} END{exit !f}'; }
while trainer_alive; do sleep 60; done
echo "[orch $(ts)] trainer python process gone" >> "$LOG"
sleep 20   # let the final save flush

if [ ! -f "$RAFT/adapter_model.safetensors" ] || [ ! -f "$RAFT/adapter_config.json" ]; then
  echo "[orch $(ts)] FAILED: adapter not found in $RAFT — training likely crashed; NOT dumping" >> "$LOG"
  rm -f "$OUT/phase2_dump.DONE"; touch "$OUT/phase2_dump.FAILED"
  exit 1
fi
echo "[orch $(ts)] adapter present ($(du -sh "$RAFT" | cut -f1)); starting 4-arm dump" >> "$LOG"

run_arm () {  # $1 arm  $2 mode  $3 adapter
  echo "[orch $(ts)] START $1 mode=$2 adapter=$3" >> "$LOG"
  "$PY" -m tools.dump_rag_preds --arm "$1" --mode "$2" --adapter "$3" \
     --out "$OUT/phase2_preds_$1.jsonl" --log "$OUT/phase2_dump_$1.armlog" >> "$LOG" 2>&1
  echo "[orch $(ts)] END $1 rc=$?" >> "$LOG"
}

run_arm R0 rag        BASE
run_arm R1 rag        "$PROMO"
run_arm R2 rag        "$RAFT"
run_arm R3 closedbook "$RAFT"

echo "[orch $(ts)] === summary ===" >> "$LOG"
ok=1
for a in R0 R1 R2 R3; do
  n=$(wc -l < "$OUT/phase2_preds_$a.jsonl" 2>/dev/null || echo 0)
  echo "  $a lines=$n" >> "$LOG"
  [ "$n" -eq 481 ] || ok=0
done
# DUMP GOTCHA: R1/R2/R3 must each have applied a LoRA adapter (R0 is BASE)
applied=$(grep -c "🔌 Applying LoRA" "$LOG")
echo "  '🔌 Applying LoRA' count=$applied (expect 3: R1,R2,R3)" >> "$LOG"
[ "$applied" -eq 3 ] || ok=0
if [ "$ok" -eq 1 ]; then
  rm -f "$OUT/phase2_dump.FAILED"; touch "$OUT/phase2_dump.DONE"
  echo "[orch $(ts)] ALL ARMS DONE + verified (4×481, LoRA×3) -> DONE marker" >> "$LOG"
else
  rm -f "$OUT/phase2_dump.DONE"; touch "$OUT/phase2_dump.FAILED"
  echo "[orch $(ts)] FAILED: line-count or LoRA-apply check did not pass -> FAILED marker" >> "$LOG"
  exit 1
fi
