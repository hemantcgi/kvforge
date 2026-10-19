# `shards/` — blind co-judge inputs

The 10 shards (`shard_00.json` … `shard_09.json`) are the **blinded, grouped-by-fact**
inputs handed to the fresh-Opus judges. Arm identities are hidden: each fact's
candidate predictions are shuffled and relabelled `A`/`B`/`C`/… , and identical
prediction strings within a fact are deduped to a single labelled candidate. The
un-blinding map lives in `../cojudge_p2_key.json` (kept **outside** this folder on
purpose, so a judge reading a shard cannot see which arm produced which answer).

Small (~364 KB total) and committed so the exact blind inputs are auditable. They
are also fully regenerable: `tools/cojudge_build_phase2.py` reproduces these
byte-for-byte from the prediction dumps (seed=42).

## Record schema (one shard = a list of facts)

```json
{
  "fact_id": "f0000",
  "question": "...",
  "gold": "...",
  "candidates": [
    {"label": "A", "prediction": "..."},
    {"label": "B", "prediction": "..."}
  ]
}
```

A judge grades each `{label: prediction}` against `gold` as CORRECT / PARTIAL /
INCORRECT; those grades are recorded in `../judge_out/shard_XX.json`.
