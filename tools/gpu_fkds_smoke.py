#!/usr/bin/env python3
"""Quick GPU smoke test for the integrated compute_fkds function."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/kvforge")

from core.config import load_config
from pipeline.prs_evaluator import compute_fkds


def main():
    uc = sys.argv[1] if len(sys.argv) > 1 else "usecase1_customer_support"
    cfg_path = f"/home/ubuntu/kvforge/examples/{uc}/config.json"
    faq_path = f"/home/ubuntu/kvforge/examples/{uc}/faqs.json"

    cfg = load_config(cfg_path).get_merged_config("indexing", "training")
    with open(faq_path) as f:
        faqs = json.load(f)

    # Use a small subset for speed: first 2 chunks with one FAQ each.
    cid_to_faq = {}
    for faq in faqs:
        for cid in faq.get("source_chunk_ids", []):
            if cid not in cid_to_faq:
                cid_to_faq[cid] = faq
            if len(cid_to_faq) >= 2:
                break
        if len(cid_to_faq) >= 2:
            break

    test_faqs = []
    seen_cids = set()
    for faq in faqs:
        overlap = set(faq.get("source_chunk_ids", [])) & set(cid_to_faq.keys())
        if overlap and not overlap & seen_cids:
            test_faqs.append(faq)
            seen_cids.update(overlap)
        if len(test_faqs) >= 2:
            break

    print(f"Running compute_fkds on {uc} with {len(test_faqs)} FAQs covering {len(seen_cids)} chunks")
    start = time.time()
    mean_kds, mean_fkds, fkds_by_chunk = compute_fkds(
        test_faqs, cfg, sample_cap=2, n=2, factual_weight=0.1
    )
    elapsed = time.time() - start
    print(f"mean_kds={mean_kds:.4f} mean_fkds={mean_fkds:.4f} elapsed={elapsed:.1f}s")
    for cid, entry in fkds_by_chunk.items():
        print(f"  {cid}: kds={entry['kds']:.4f} factual={entry['factual_accuracy']:.4f} fkds={entry['fkds']:.4f}")


if __name__ == "__main__":
    main()
