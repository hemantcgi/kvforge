#!/usr/bin/env python3
"""RAFT step 2-3: assemble faqs_raft.json from the 60k QA + retrieval table,
and evaluate the §3 coverage gate. Thin I/O wrapper around the unit-tested
tools/raft_data.py (build_examples / dedup_against_eval / filter_481 / split).

Usage (run from repo root):
  python tools/build_raft_trainset.py \
    --retrieval retrieval_60k.json \
    --chunks   examples/usecase4_bedrock_userguide/chunks.json \
    --eval     examples/usecase4_bedrock_userguide/eval_500.json \
    --recall   results/knowledge/uc4/recall_at_k_top5.json \
    --out      examples/usecase4_bedrock_userguide/faqs_raft.json \
    --stats    results/knowledge/uc4/raft_build_stats.json \
    --p 0.8 --n-context 5 --seed 42
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raft_data as R


def _load(path):
    return json.loads(open(path).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True, help="retrieval_60k.json [{question,answer,retrieved_ids}]")
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--eval", required=True, help="eval_500.json")
    ap.add_argument("--recall", required=True, help="recall_at_k_top5.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--p", type=float, default=0.8)
    ap.add_argument("--n-context", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gold-gate", type=float, default=0.80)
    ap.add_argument("--miss-gate", type=float, default=0.70)
    a = ap.parse_args()

    chunks = _load(a.chunks)
    id2text = {int(c["chunk_id"]): c["text"] for c in chunks}
    chunks_norm = R.attach_norm([{"chunk_id": int(c["chunk_id"]), "text": c["text"]} for c in chunks])
    print(f"chunks: {len(id2text)}", flush=True)

    records = _load(a.retrieval)
    print(f"retrieval records: {len(records)}", flush=True)

    ev = _load(a.eval)
    eval_items = ev["items"] if isinstance(ev, dict) and "items" in ev else ev

    # leak control: dedup 60k training QA vs eval_500 (all 500)
    kept, removed = R.dedup_against_eval(records, eval_items, jaccard_thresh=0.80)
    print(f"leak-dedup: kept {len(kept)}, removed {len(removed)}", flush=True)

    examples, stats = R.build_examples(kept, id2text, chunks_norm,
                                       p=a.p, n_context=a.n_context, seed=a.seed)
    trained = stats["trained_oracle_ids"]

    # §3 coverage gate — fact TRAINED (chunk is a recovered oracle of >=1 example)
    f481 = R.filter_481(eval_items)
    gold_481 = {int(r["chunk_id"]) for r in f481 if r.get("chunk_id") is not None}
    recall = _load(a.recall)
    by_gold = {}
    for rr in (recall.get("records") if isinstance(recall, dict) else recall) or []:
        g = rr.get("gold_chunk_id", rr.get("chunk_id"))
        if g is not None:
            by_gold[int(g)] = bool(rr.get("hit"))
    miss = {c for c in gold_481 if by_gold.get(c) is False}

    gold_cov = len(gold_481 & trained) / max(1, len(gold_481))
    miss_cov = len(miss & trained) / max(1, len(miss))
    gate_pass = gold_cov >= a.gold_gate and miss_cov >= a.miss_gate

    json.dump([{"question": e["question"], "answer": e["answer"]} for e in examples],
              open(a.out, "w"))
    report = {
        "n_examples": stats["n_examples"],
        "leak_removed": len(removed),
        "recovery_rate": stats["recovery_rate"],
        "recovered": stats["recovered"],
        "withheld": stats["withheld"],
        "withheld_answer_leak_rate": stats["withheld_answer_leak_rate"],
        "p_keep_oracle": a.p, "n_context": a.n_context, "seed": a.seed,
        "gold_481": len(gold_481), "miss_191": len(miss),
        "gold_coverage": round(gold_cov, 4), "miss_coverage": round(miss_cov, 4),
        "gold_gate": a.gold_gate, "miss_gate": a.miss_gate,
        "GATE_PASS": gate_pass,
        "distinct_trained_oracle_chunks": len(trained),
    }
    os.makedirs(os.path.dirname(a.stats), exist_ok=True)
    json.dump({**report, "trained_oracle_ids_sorted": sorted(trained)},
              open(a.stats, "w"), indent=1)
    print(json.dumps(report, indent=1), flush=True)
    print(("✅ GATE PASS" if gate_pass else "⛔ GATE FAIL") +
          f"  gold {gold_cov:.3f}>={a.gold_gate}  miss {miss_cov:.3f}>={a.miss_gate}", flush=True)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    sys.exit(main())
