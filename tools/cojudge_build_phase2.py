#!/usr/bin/env python3
"""Build blind grouped-by-fact co-judge shards for RAFT Phase-2 (5 arms).

Arms: R0/R1/R2 = RAG top-5 (base/promo/raft); R3 = RAFT closed-book;
R1cb = promo closed-book (prior deterministic dump, re-judged fresh in-session).
Shards (blinded, no arm labels) -> <out-dir>/shards/ ; key kept OUTSIDE that dir
(<out-dir>/cojudge_p2_key.json). Also stashes per-fact hit/miss + answer_in_context
+ leak flag (<out-dir>/cojudge_p2_facts_meta.json) for scoring.

Inputs:
  --preds-dir : phase2_preds_{R0,R1,R2,R3,R1cb}.jsonl
  --train     : the packed RAFT training set (faqs_full_idea1.json / faqs_raft.json),
                used ONLY to compute the per-fact leak flag. Large (~9.5 MB) — kept in
                the GPU host backup, not committed. The resulting meta (with leak flags) IS
                committed, so scoring does not need this file.

The downstream judging step (fresh Opus grading each blind shard) is NOT
deterministic, so the checked-in shards/key/meta under
results/knowledge/uc4/cojudge_phase2/ are the canonical judged artifacts; this
script documents and regenerates the blinding, not the grading.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import cojudge_phase2 as CJ  # noqa: E402

ARMS = ["R0", "R1", "R2", "R3", "R1cb"]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-dir", default=os.path.join(ROOT, "results/knowledge/uc4/cojudge_phase2/preds"),
                    help="dir holding the FULL phase2_preds_{R0,R1,R2,R3,R1cb}.jsonl dumps "
                         "(large, not committed — regenerate with tools/dump_rag_preds.py or use the "
                         "GPU host backup; the committed preds/ holds only samples). See preds/README.md")
    ap.add_argument("--train", required=True,
                    help="packed RAFT training set for leak-flag computation (large; in the GPU host backup)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results/knowledge/uc4/cojudge_phase2"),
                    help="output dir for shards/, cojudge_p2_key.json, cojudge_p2_facts_meta.json")
    ap.add_argument("--n-shards", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = {a: os.path.join(args.preds_dir, f"phase2_preds_{a}.jsonl") for a in ARMS}

    def load(f):
        return [json.loads(line) for line in open(f)]

    rows_by_arm = {a: {str(r["chunk_id"]): r for r in load(files[a])} for a in ARMS}
    preds_by_arm = {
        a: {c: {"question": r["question"], "gold": r["gold"], "prediction": r["prediction"]}
            for c, r in rows_by_arm[a].items()}
        for a in ARMS
    }

    records, key = CJ.build_blind_facts(preds_by_arm, ARMS, seed=args.seed)
    assert len(records) == 481, f"expected 481 facts, got {len(records)}"

    # ---- dedup stats ----
    ncand = Counter(len(r["candidates"]) for r in records)
    total_items = sum(len(r["candidates"]) for r in records)
    print(f"facts={len(records)}  judge items (deduped)={total_items}  "
          f"vs naive {len(records) * len(ARMS)}")
    print(f"unique-candidates-per-fact distribution: {dict(sorted(ncand.items()))}")

    # ---- per-fact meta for scoring: hit/miss (from R0 rag row), answer_in_context, leak ----
    # leak criterion reused verbatim from the Phase-1 promotion leak split.
    train = json.load(open(args.train))
    train_q = "\n".join(norm(r["question"]) for r in train)
    train_a = "\n".join(norm(r["answer"]) for r in train)

    def leaked(q, gold):
        nq, na = norm(q), norm(gold)
        return (len(nq) >= 25 and nq in train_q) or (len(na) >= 25 and na in train_a)

    meta = {}
    for fid, k in key.items():
        cid = k["chunk_id"]
        r0 = rows_by_arm["R0"][cid]
        meta[fid] = {
            "chunk_id": cid,
            "hit": bool(r0["hit"]),
            "answer_in_context": bool(r0["answer_in_context"]),
            "leaked": leaked(k["question"], k["gold"]),
        }
    n_hit = sum(m["hit"] for m in meta.values())
    n_leak = sum(m["leaked"] for m in meta.values())
    aic_miss = sum(1 for m in meta.values() if m["answer_in_context"] and not m["hit"])
    print(f"hit={n_hit} miss={481 - n_hit}  leaked={n_leak} clean={481 - n_leak}  "
          f"answer_in_context_on_misses={aic_miss}")
    # assert the promotion split reproduces (clean n=406, leaked 75)
    assert n_hit == 290, f"hit split changed: {n_hit}"
    assert n_leak == 75, f"leak split != promotion's 75: got {n_leak}"

    # ---- write shards (blinded) + key (separate dir) + meta ----
    shard_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    shards = CJ.shard_records(records, args.n_shards)
    for i, s in enumerate(shards):
        json.dump(s, open(os.path.join(shard_dir, f"shard_{i:02d}.json"), "w"), indent=1)
    json.dump(key, open(os.path.join(args.out_dir, "cojudge_p2_key.json"), "w"), indent=1)   # NOT in shard dir
    json.dump(meta, open(os.path.join(args.out_dir, "cojudge_p2_facts_meta.json"), "w"), indent=1)
    print(f"shard sizes: {[len(s) for s in shards]}")
    print(f"wrote {args.n_shards} shards -> {shard_dir}; key + meta -> {args.out_dir}")


if __name__ == "__main__":
    main()
