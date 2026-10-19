# `preds/` — per-arm prediction dumps (large data: NOT committed)

The full per-arm prediction dumps are **generated data** and are deliberately
**not checked in** (they total ~5.6 MB; `R0`/`R1`/`R2` carry the full top-5
retrieved context in every record). This folder ships only:

- `samples/phase2_preds_<arm>.sample.jsonl` — the first 3 records of each arm, so
  the schema and content are inspectable without the full files.

## Arms

| File (full, not committed) | Arm | Description |
|---|---|---|
| `phase2_preds_R0.jsonl` | R0 | base Gemma-4-E2B-it + top-5 RAG |
| `phase2_preds_R1.jsonl` | R1 | promoted adapter + top-5 RAG |
| `phase2_preds_R2.jsonl` | R2 | RAFT adapter + top-5 RAG |
| `phase2_preds_R3.jsonl` | R3 | RAFT adapter, closed-book (no retrieval) |
| `phase2_preds_R1cb.jsonl` | R1cb | promoted adapter, closed-book (prior deterministic dump; renamed from `rwscale_preds_promo_allLinear.jsonl`) |

Each file has 481 JSONL records (one per eval fact).

## Record schema

```json
{
  "arm": "R0",
  "chunk_id": 30,
  "question": "...",
  "prompt": "...",              // the full rendered prompt (top-5 context for R0/R1/R2; bare question for R3/R1cb)
  "prediction": "...",          // the model's generated answer (cap 48 new tokens, chat SFT format)
  "retrieved_ids": [.. 5 ids ..], // top-5 retrieved chunk ids (empty for closed-book R3/R1cb)
  "hit": false,                 // gold chunk in top-5 (null for closed-book)
  "answer_in_context": true,    // gold answer text present in a retrieved chunk
  "gold": "..."                 // ground-truth answer
}
```

## How to regenerate the full dumps

Requires a GPU (A10G), the base model, and the R1/R2 adapters. Run one arm per
process, sequentially (single-GPU VRAM); see memory `gpu-generation-sequential`.

```bash
# from the repo root, with the repo on PYTHONPATH
python tools/dump_rag_preds.py --arm R0 --out phase2_preds_R0.jsonl                 # base + top-5
python tools/dump_rag_preds.py --arm R1 --adapter <promo_adapter>  --out phase2_preds_R1.jsonl
python tools/dump_rag_preds.py --arm R2 --adapter <raft_adapter>   --out phase2_preds_R2.jsonl
python tools/dump_rag_preds.py --arm R3 --adapter <raft_adapter>   --out phase2_preds_R3.jsonl --closedbook
```

The top-5 context is read from `results/knowledge/uc4/recall_at_k_top5.json`
(threshold=None top-5 over `bedrock-userguide-6154`), so the `hit` split
(290/191) is fixed by construction and no live Qdrant is needed at dump time.
See `tools/dump_rag_preds.py` for flags.

## Who reads these

- `tools/cojudge_build_phase2.py` — needs the full dumps to build the blind shards.
- `tools/cojudge_score_phase2.py` — **does not need preds** (the pre-registered
  verdict is computed from `../judge_out/`, `../cojudge_p2_key.json`, `../cojudge_p2_facts_meta.json`).
- `tools/cojudge_score_extended_phase2.py` — uses the full dumps only for the two
  descriptive fields `qgold_parity` and `format_parity_median_pred_words`; when the
  full dumps are absent it skips just those two and still computes every
  grade-derived field. Point `--preds-dir` at a folder holding the regenerated
  dumps (or the EC2 backup) to produce all fields.
