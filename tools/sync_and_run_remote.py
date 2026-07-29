"""Sync the local KVForge repo to the remote GPU host and run experiments.

Usage:

    python tools/sync_and_run_remote.py \
        --host 13.217.195.243 \
        --user ubuntu \
        --key /Users/hemant/Downloads/RoPE/g5.x.pem \
        --remote-path /home/ubuntu/kvforge \
        --experiments e1,e2,e3,e5 \
        --judge-api-key $OPENAI_API_KEY

The script:
1. rsyncs the current repo (excluding .git, venv, pycache, etc.) to the remote.
2. SSHs in and runs the real experiment orchestrator.
3. rsyncs the results back to docs/scientific_revision_real/.

Run one use-case at a time:

    python tools/sync_and_run_remote.py --uc-config examples/usecase3_squad/config.json ...
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], check: bool = True):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=check)


def main():
    parser = argparse.ArgumentParser(description="Sync to remote GPU and run experiments")
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--key", required=True)
    parser.add_argument("--remote-path", default="/home/ubuntu/kvforge")
    parser.add_argument("--uc-config", help="Run a single use-case")
    parser.add_argument("--all", action="store_true", help="Run all use-cases")
    parser.add_argument("--experiments", default="e1,e2,e3,e5")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--judge-provider", default="openai")
    parser.add_argument("--judge-api-key", default="")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--local-results", default="docs/scientific_revision_real")
    args = parser.parse_args()

    ssh = ["ssh", "-i", args.key, "-o", "StrictHostKeyChecking=no", f"{args.user}@{args.host}"]
    rsync_ssh = f"-e ssh -i {args.key} -o StrictHostKeyChecking=no"
    rsync_excludes = [
        "--exclude=.git", "--exclude=.worktrees", "--exclude=venv",
        "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=.pytest_cache",
        "--exclude=docs/scientific_revision", "--exclude=docs/scientific_revision_real",
        "--exclude=*.pdf", "--exclude=*.aux", "--exclude=*.out", "--exclude=*.log",
        # Preserve remote-generated artifacts (datasets, checkpoints, state, logs)
        "--exclude=examples/*/data/**",
        "--exclude=examples/*/lora_checkpoints/**",
        "--exclude=examples/*/version.json",
        "--exclude=examples/*/.chroma/**",
        "--exclude=examples/*/.chroma",
        "--exclude=examples/*/.faiss/**",
        "--exclude=examples/*/.faiss",
        "--exclude=examples/*/.qdrant/**",
        "--exclude=examples/*/.qdrant",
        "--exclude=examples/*/.qdrant_storage/**",
        "--exclude=examples/*/.qdrant_storage",
        "--exclude=examples/*/qdrant_data/**",
        "--exclude=examples/*/qdrant_data",
        "--exclude=.chroma/**",
        "--exclude=.chroma",
        "--exclude=.faiss/**",
        "--exclude=.faiss",
        "--exclude=.qdrant/**",
        "--exclude=.qdrant",
        "--exclude=*.db",
        "--exclude=*.sqlite3",
        "--exclude=*.sqlite3-journal",
        "--exclude=*.safetensors",
        "--exclude=*.bin",
        "--exclude=*.pt",
        "--exclude=*.pth",
    ]

    # 1. Sync code to remote (do not delete remote-only artifacts)
    run(["rsync", "-avz", "--delete", rsync_ssh] + rsync_excludes + [
        str(ROOT) + "/", f"{args.user}@{args.host}:{args.remote_path}/"
    ])

    # 2. Run experiments on remote
    remote_cmd = f"cd {args.remote_path} && source venv/bin/activate && python tools/run_real_experiments.py"
    if args.all:
        remote_cmd += " --all"
    elif args.uc_config:
        remote_cmd += f" --uc-config {args.uc_config}"
    else:
        raise ValueError("Specify --uc-config or --all")
    remote_cmd += f" --experiments {args.experiments}"
    remote_cmd += f" --output {args.local_results}"
    remote_cmd += f" --judge-provider {args.judge_provider}"
    remote_cmd += f" --judge-api-key {args.judge_api_key}"
    remote_cmd += f" --judge-model {args.judge_model}"
    if args.max_samples:
        remote_cmd += f" --max-samples {args.max_samples}"

    run(ssh + [remote_cmd])

    # 3. Sync results back
    local_results = ROOT / args.local_results
    local_results.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-avz", f"{args.user}@{args.host}:{args.remote_path}/{args.local_results}/", str(local_results) + "/"])

    print(f"\n✓ Results are in {local_results}")


if __name__ == "__main__":
    main()
