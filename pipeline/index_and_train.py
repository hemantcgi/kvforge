"""
index_and_train.py — Orchestrator: for each new document, run SP1 → SP2 → KV refresh.

Usage:
  python3 -m pipeline.index_and_train new_document.pdf
  python3 -m pipeline.index_and_train new_document.pdf --config my_config.json --skip-prs
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'─'*60}\n▶  {desc}\n{'─'*60}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print(f"❌ Failed: {desc}")
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pdf_file")
    p.add_argument("--config", default="my_config.json")
    p.add_argument("--replay-ratio", type=float, default=0.2)
    p.add_argument("--skip-prs", action="store_true",
                   help="Skip PRS evaluation (faster, use during testing)")
    p.add_argument("--faqs", default="bedrock_50 faqs.json")
    args = p.parse_args()

    pdf = Path(args.pdf_file)
    if not pdf.exists():
        print(f"❌ File not found: {pdf}")
        sys.exit(1)

    py = sys.executable

    # ── Step 1: Index (chunk + embed + KV tensors) ─────────────────────────
    run([py, "-m", "pipeline.kv_indexer", "--config", args.config, "index", str(pdf)],
        f"SP1: Index {pdf.name}")

    # ── Step 2: LoRA fine-tune ─────────────────────────────────────────────
    run([py, "-m", "pipeline.lora_trainer",
         "--config", args.config,
         "--source-file", pdf.name,
         "--replay-ratio", str(args.replay_ratio)],
        "SP2: LoRA fine-tune")

    # ── Step 3: Recompute KV for new chunks with updated weights ───────────
    run([py, "-m", "pipeline.kv_indexer", "--config", args.config,
         "compute-kv", "--source-file", pdf.name],
        "SP1: Recompute KV for new chunks with updated weights")

    # ── Step 4: Proactively heal ALL stale-versioned chunks ─────────────────
    # Reads current_lora_version from version.json and heals all chunks whose
    # kv_version < N (previously versioned by an earlier LoRA round).
    with open(args.config) as _f:
        _cfg = json.load(_f)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import version as _ver
    _ver.init(_cfg)
    current_ver = _ver.get_lora_version()
    if current_ver > 0:
        run([py, "-m", "pipeline.kv_indexer", "--config", args.config,
             "compute-kv", "--stale-version", str(current_ver)],
            f"SP1: Proactive KV heal for all stale chunks (< v{current_ver})")

    # ── Step 5: PRS evaluation ─────────────────────────────────────────────
    if not args.skip_prs:
        run([py, "-m", "pipeline.prs_evaluator",
             "--config", args.config,
             "--faqs", args.faqs],
            "SP2: PRS evaluation")

    # When --skip-prs is set, advance manually since append_prs is not called.
    # When PRS runs normally, prs_evaluator calls append_prs which handles advancement.
    if args.skip_prs:
        _ver.activate_phase_2()

    print(f"\n✅ index_and_train complete for {pdf.name}")
    print("   Stale chunks from prior rounds have been healed proactively.")


if __name__ == "__main__":
    main()
