#!/usr/bin/env python3
"""
setup.py — Download and prepare the PubMedQA biomedical dataset.

Downloads 'qiaojin/PubMedQA' (pqa_labeled split) from HuggingFace, saves:
  - data/corpus.jsonl : context paragraphs as {"text": "...", "source": "..."}
  - faqs.json         : 50 Q&A pairs as {"question": "...", "answer": "..."}

Run from repo root:
    python examples/usecase2_pubmedqa/setup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
FAQS_PATH = HERE / "faqs.json"

DATASET_NAME = "qiaojin/PubMedQA"
DATASET_CONFIG = "pqa_labeled"
CORPUS_SIZE = 1_000   # context paragraphs
FAQ_SIZE = 50


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {DATASET_NAME} ({DATASET_CONFIG}) …")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", trust_remote_code=True)
    print(f"  {len(ds)} records available")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Corpus: flatten context paragraphs ────────────────────────────────────
    written = 0
    with open(CORPUS_PATH, "w") as f:
        for row in ds:
            if written >= CORPUS_SIZE:
                break
            # context["contexts"] is a list of paragraph strings
            contexts = row.get("context", {}).get("contexts", [])
            labels = row.get("context", {}).get("labels", [""] * len(contexts))
            pubmed_id = str(row.get("pubid", "unknown"))
            for para, label in zip(contexts, labels):
                para = para.strip()
                if len(para.split()) < 20:
                    continue
                record = {
                    "text": para,
                    "source": pubmed_id,
                    "relevance": label,
                }
                f.write(json.dumps(record) + "\n")
            written += 1
    print(f"  Corpus written: {CORPUS_PATH}  ({written} records → multiple paragraphs each)")

    # ── FAQs: question + long_answer pairs ────────────────────────────────────
    faqs: list[dict] = []
    for row in ds:
        if len(faqs) >= FAQ_SIZE:
            break
        q = row.get("question", "").strip()
        a = row.get("long_answer", "").strip()
        if len(q) < 10 or len(a) < 10:
            continue
        faqs.append({"question": q, "answer": a})

    FAQS_PATH.write_text(json.dumps(faqs, indent=2))
    print(f"  FAQs written:   {FAQS_PATH}  ({len(faqs)} pairs)")
    print("\nSetup complete. Now run: bash examples/usecase2_pubmedqa/run_pipeline.sh")


if __name__ == "__main__":
    main()
