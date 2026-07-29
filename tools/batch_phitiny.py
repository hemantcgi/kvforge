"""Batch queue remaining Phi-tiny-MoE experiments.
Run after 2WikiMQA 5x training completes to chain UC4 5x + evals.
"""
from __future__ import annotations

import subprocess, sys, time, os
from pathlib import Path

BASE = Path("/home/ubuntu/kvforge")
VENV = BASE / "venv" / "bin" / "python3"
LOGS = BASE / "logs"

cmds = []

# 1. UC4 5x training (after 2WikiMQA 5x finishes)
cmds.append((
    "UC4 5x training",
    [
        str(VENV), "-u", "tools/train_phitiny.py",
        "--qa-pairs", "examples/usecase4_bedrock_userguide/diversified_v1/5x/qa_pairs.json",
        "--output", "lora_checkpoints/phitiny_uc4_5x_v1",
        "--epochs", "1",
    ],
    str(LOGS / "train_phitiny_uc4_5x.log"),
))

# 2. UC4 1x eval
cmds.append((
    "UC4 1x eval",
    [
        str(VENV), "-u", "tools/eval_phitiny.py",
        "--checkpoint", "lora_checkpoints/phitiny_uc4_1x_v1",
        "--output", "results/pathb_diversity/uc4/phitiny_path_b_1x",
        "--eval-set", "examples/usecase4_bedrock_userguide/eval_heldout_v1.json",
    ],
    str(LOGS / "eval_phitiny_uc4_1x.log"),
))

# 3. UC4 5x eval
cmds.append((
    "UC4 5x eval",
    [
        str(VENV), "-u", "tools/eval_phitiny.py",
        "--checkpoint", "lora_checkpoints/phitiny_uc4_5x_v1",
        "--output", "results/pathb_diversity/uc4/phitiny_path_b_5x",
        "--eval-set", "examples/usecase4_bedrock_userguide/eval_heldout_v1.json",
    ],
    str(LOGS / "eval_phitiny_uc4_5x.log"),
))

# 4. 2WikiMQA 1x eval
cmds.append((
    "2WikiMQA 1x eval",
    [
        str(VENV), "-u", "tools/eval_phitiny.py",
        "--checkpoint", "lora_checkpoints/phitiny_2wikimqa_1x_v1",
        "--output", "results/pathb_diversity/2wikimqa/phitiny_path_b_1x",
        "--eval-set", "examples/longbench_2wikimqa/eval_2wikimqa.json",
    ],
    str(LOGS / "eval_phitiny_2wikimqa_1x.log"),
))

# 5. 2WikiMQA 5x eval
cmds.append((
    "2WikiMQA 5x eval",
    [
        str(VENV), "-u", "tools/eval_phitiny.py",
        "--checkpoint", "lora_checkpoints/phitiny_2wikimqa_5x_v1",
        "--output", "results/pathb_diversity/2wikimqa/phitiny_path_b_5x",
        "--eval-set", "examples/longbench_2wikimqa/eval_2wikimqa.json",
    ],
    str(LOGS / "eval_phitiny_2wikimqa_5x.log"),
))

# Wait until 2WikiMQA 5x finishes before starting
TRAIN_PID_FILE = Path("/tmp/phitiny_5x_train_pid")

def is_5x_done():
    if not TRAIN_PID_FILE.exists():
        # Also check if the checkpoint exists
        return (BASE / "lora_checkpoints" / "phitiny_2wikimqa_5x_v1" / "adapter_model.safetensors").exists()
    pid = int(TRAIN_PID_FILE.read_text().strip())
    try:
        os.kill(pid, 0)
        return False
    except OSError:
        return True

print("Waiting for 2WikiMQA 5x training to complete...")
while not is_5x_done():
    sys.stdout.flush()
    time.sleep(60)

print("2WikiMQA 5x done! Starting remaining experiments...")

for name, cmd, logfile in cmds:
    print(f"\n{'='*60}")
    print(f"Starting: {name}")
    print(f"{'='*60}")
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "w") as f:
        proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=f, stderr=subprocess.STDOUT)
        rc = proc.wait()
    if rc != 0:
        print(f"⚠️ {name} FAILED (rc={rc})")
    else:
        print(f"✅ {name} complete")

print(f"\n{'='*60}")
print("All experiments complete!")
print(f"{'='*60}")
