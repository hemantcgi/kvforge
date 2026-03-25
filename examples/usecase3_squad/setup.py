#!/usr/bin/env python3
"""
setup.py — Download and prepare the SQuAD v2 dataset.

Downloads 'rajpurkar/squad_v2' from HuggingFace, saves:
  - data/corpus.jsonl : unique Wikipedia passage contexts as {"text": "...", "source": "..."}
  - faqs.json         : 50 answerable Q&A pairs as {"question": "...", "answer": "..."}

SQuAD v2 contains ~130 000 questions from Wikipedia passages. We use unique
passage contexts as the knowledge corpus and answerable questions for PRS
evaluation (unanswerable questions are skipped).

Run from repo root:
    python examples/usecase3_squad/setup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
FAQS_PATH = HERE / "faqs.json"

DATASET_NAME = "rajpurkar/squad_v2"
CORPUS_SIZE = 2_000   # unique passage contexts
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

    # ── Corpus: unique passage contexts (de-duplicated) ───────────────────────
    seen_contexts: set[str] = set()
    faqs: list[dict] = []
    written = 0

    with open(CORPUS_PATH, "w") as f:
        for row in ds:
            context = row["context"].strip()
            question = row["question"].strip()
            answers = row.get("answers", {}).get("text", [])

            # Corpus: unique passages only
            if context not in seen_contexts and written < CORPUS_SIZE:
                seen_contexts.add(context)
                record = {
                    "text": context,
                    "source": row.get("title", "wikipedia").replace(" ", "_"),
                }
                f.write(json.dumps(record) + "\n")
                written += 1

            # FAQs: answerable questions only (non-empty answers list)
            if len(faqs) < FAQ_SIZE and answers:
                answer_text = answers[0].strip()
                if len(question) >= 10 and len(answer_text) >= 2:
                    faqs.append({"question": question, "answer": answer_text})

    print(f"  Corpus written: {CORPUS_PATH}  ({written} unique passages)")

    FAQS_PATH.write_text(json.dumps(faqs, indent=2))
    print(f"  FAQs written:   {FAQS_PATH}  ({len(faqs)} pairs)")
    print("\nSetup complete. Now run: bash examples/usecase3_squad/run_pipeline.sh")


if __name__ == "__main__":
    main()
