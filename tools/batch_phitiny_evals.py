"""Run Phi-tiny-MoE evals after UC4 5x training completes.

Usage: python3 tools/batch_phitiny_evals.py
"""
from __future__ import annotations

import json, subprocess, sys, time, os
from pathlib import Path

BASE = Path("/home/ubuntu/kvforge")
VENV = str(BASE / "venv" / "bin" / "python3")
LOGS = BASE / "logs"

evals = [
    ("UC4 1x", "lora_checkpoints/phitiny_uc4_1x_v1",
     "results/pathb_diversity/uc4/phitiny_path_b_1x",
     "examples/usecase4_bedrock_userguide/eval_heldout_v1.json"),
    ("2WikiMQA 1x", "lora_checkpoints/phitiny_2wikimqa_1x_v1",
     "results/pathb_diversity/2wikimqa/phitiny_path_b_1x",
     "examples/longbench_2wikimqa/eval_2wikimqa.json"),
    ("2WikiMQA 5x", "lora_checkpoints/phitiny_2wikimqa_5x_v1",
     "results/pathb_diversity/2wikimqa/phitiny_path_b_5x",
     "examples/longbench_2wikimqa/eval_2wikimqa.json"),
    ("UC4 5x (after training)", "lora_checkpoints/phitiny_uc4_5x_v1",
     "results/pathb_diversity/uc4/phitiny_path_b_5x",
     "examples/usecase4_bedrock_userguide/eval_heldout_v1.json"),
]

# Wait for UC4 5x training to finish
UC4_5X_CKPT = BASE / "lora_checkpoints" / "phitiny_uc4_5x_v1" / "adapter_model.safetensors"
print("Waiting for UC4 5x training to complete...")
while not UC4_5X_CKPT.exists():
    time.sleep(120)

# Give it a moment to finish writing
time.sleep(30)

for name, ckpt, output, eval_set in evals:
    out_dir = BASE / output
    summary_file = out_dir / "summary.json"
    if summary_file.exists():
        print(f"Skipping {name} — already evaluated at {output}")
        continue
    logfile = LOGS / f"eval_phitiny_{name.lower().replace(' ','_').replace('(','').replace(')','')}.log"
    cmd = [VENV, "-u", "tools/eval_phitiny.py",
           "--checkpoint", ckpt,
           "--output", output,
           "--eval-set", eval_set]
    print(f"\n{'='*60}")
    print(f"Eval: {name}")
    print(f"{'='*60}")
    with open(logfile, "w") as f:
        rc = subprocess.Popen(cmd, cwd=str(BASE), stdout=f, stderr=subprocess.STDOUT).wait()
    if rc != 0:
        print(f"  FAILED (rc={rc})")
    else:
        s = json.loads(summary_file.read_text())
        fkds = s["modes"]["parametric"]["fkds"]
        print(f"  fKDS: {fkds['mean']:.4f} ± {fkds['sem']:.4f}  (n={fkds['n']})")

print(f"\n{'='*60}")
print("All evals complete!")
print(f"{'='*60}")
