#!/usr/bin/env python3
"""
setup.py — Download and prepare the Bitext customer-support dataset.

Downloads 'bitext/Bitext-customer-support-llm-chatbot-training-dataset'
from HuggingFace, saves:
  - data/corpus.jsonl   : 2 000 records as {"text": "...", "source": "..."}
  - faqs.json           : 50 Q&A pairs as {"question": "...", "answer": "..."}

Run from repo root:
    python examples/usecase1_customer_support/setup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
FAQS_PATH = HERE / "faqs.json"

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
CORPUS_SIZE = 2_000
FAQ_SIZE = 50


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {DATASET_NAME} …")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"  {len(ds)} records available")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Corpus: instruction + response concatenated as a text chunk ──────────
    written = 0
    with open(CORPUS_PATH, "w") as f:
        for row in ds:
            if written >= CORPUS_SIZE:
                break
            text = (
                f"Customer: {row['instruction'].strip()}\n"
                f"Support:  {row['response'].strip()}"
            )
            record = {
                "text": text,
                "source": f"{row.get('category', 'general')}_{row.get('intent', 'unknown')}",
            }
            f.write(json.dumps(record) + "\n")
            written += 1
    print(f"  Corpus written: {CORPUS_PATH}  ({written} records)")

    # ── FAQs: use instruction as question, response as answer ────────────────
    faqs: list[dict] = []
    seen: set[str] = set()
    for row in ds:
        if len(faqs) >= FAQ_SIZE:
            break
        q = row["instruction"].strip()
        a = row["response"].strip()
        if q in seen or len(q) < 10 or len(a) < 10:
            continue
        seen.add(q)
        faqs.append({"question": q, "answer": a})

    FAQS_PATH.write_text(json.dumps(faqs, indent=2))
    print(f"  FAQs written:   {FAQS_PATH}  ({len(faqs)} pairs)")
    print("\nSetup complete. Now run: bash examples/usecase1_customer_support/run_pipeline.sh")


if __name__ == "__main__":
    main()
