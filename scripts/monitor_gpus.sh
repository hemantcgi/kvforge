#!/usr/bin/env bash
# KVForge GPU monitor — check status and auto-schedule next queued job.
# Usage: bash scripts/monitor_gpus.sh [--launch]
#   --launch: auto-launch next queued job when GPU is free
set -e

EC2_PEM="/Users/hemant/Downloads/RoPE/g5.x.pem"
EC2_A="ubuntu@32.199.183.61"
EC2_B="ubuntu@18.234.222.244"
AUTO_LAUNCH=false
[[ "$1" == "--launch" ]] && AUTO_LAUNCH=true

RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m' NC='\033[0m'

check_gpu() {
    local label="$1" host="$2"
    echo -e "${GREEN}=== $label ($host) ===${NC}"
    local gpu_info=$(ssh -i "$EC2_PEM" -o ConnectTimeout=5 "$host" \
        'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader' 2>/dev/null)
    if [[ -z "$gpu_info" ]]; then
        echo -e "  ${RED}UNREACHABLE${NC}"
        return 1
    fi
    echo "$gpu_info" | while IFS=, read -r idx name mem_used_raw mem_total_raw util; do
        local mem_used=$(echo "$mem_used_raw" | tr -d -c 0-9)
        local mem_total=$(echo "$mem_total_raw" | tr -d -c 0-9)
        local util_pct=$(echo "$util" | tr -d -c 0-9)
        status="BUSY"
        [[ $mem_used -lt 100 ]] && status="${GREEN}FREE${NC}"
        printf "  GPU%-2d %-40s %5d/%5d MiB  %3d%%  %s\n" "$idx" "$name" "$mem_used" "$mem_total" "$util_pct" "$status"
    done
    return 0
}

check_jobs() {
    local host="$1"
    ssh -i "$EC2_PEM" -o ConnectTimeout=5 "$host" \
        'ps -eo pid,stat,comm --no-header 2>/dev/null' 2>/dev/null | \
        grep -E "measure_baseline|lora_train|python" | grep -v grep | head -5
}

check_results() {
    local host="$1"
    ssh -i "$EC2_PEM" -o ConnectTimeout=5 "$host" '
        cd ~/kvforge 2>/dev/null || exit 0
        for d in results/phase3_tiers_* results/ablation_prompt_* results/fulltoken_sweep_*; do
            [[ -d "$d" ]] || continue
            echo "  $(basename "$d"):"
            for f in "$d"/*/summary.json; do
                [[ -f "$f" ]] || continue
                local tier=$(basename "$(dirname "$f")")
                python3 -c "
import json; d=json.load(open(\"$f\")); m=d[\"modes\"]
for mode in text_rag kv_meanpool kv_fulltoken parametric system_prompt simple chat_template; do
    v=m.get(mode,{})\n    fa=v.get(\"factual_accuracy\",{})\n    if fa: print(f\"     {tier}/{mode} fa={fa.get(\"mean\",0):.4f}\")
" 2>/dev/null
            done 2>/dev/null
        done
    ' 2>/dev/null
}

launch_next() {
    local host="$1" gpu="$2"
    echo -e "${YELLOW}Queue: check experiment_checklist.md for next job on $host GPU$gpu${NC}"
    echo "  Run: cd ~/kvforge && source venv/bin/activate && CUDA_VISIBLE_DEVICES=$gpu <command> > logs/job.log 2>&1 &"
    # TODO: auto-read checklist and launch
}

# ── Main ──────────────────────────────────────────────────
check_gpu "EC2-A" "$EC2_A"
echo ""
check_gpu "EC2-B" "$EC2_B"
echo ""

echo -e "${GREEN}=== Running Jobs (EC2-B) ===${NC}"
check_jobs "$EC2_B"

echo ""
echo -e "${GREEN}=== Recent Results ===${NC}"
check_results "$EC2_B"

echo ""
echo -e "${GREEN}=== Next Queued ===${NC}"
grep -A 6 "GPU ALLOCATION" ~/kvforge/docs/experiment_checklist.md 2>/dev/null | head -7 || \
    grep -A 6 "GPU ALLOCATION" "$(dirname "$0")/../docs/experiment_checklist.md" 2>/dev/null | head -7
