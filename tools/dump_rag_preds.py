"""RAG-mode 4-arm prediction dump for RAFT Phase-2 (plan §5.5). NO judge here.

Per 481 eval fact (eval_500 minus \\bpage\\b trivia):
  R0/R1/R2 (mode=rag): context = the raw top-5 retrieved chunks
      (`recall_at_k_top5.json.retrieved_ids` — exactly the threshold=None top-5
      from `bedrock-userguide-6154`, so `hit` matches the 290/191 split by
      construction and no live Qdrant is needed at dump time), rendered via the
      SHARED `render_prompt` so train/eval prompts are byte-identical (§5.2).
  R3 (mode=closedbook): question only (OOD for R2 — report as directional).

Generate at cap 48, chat SFT format. One arm per process so VRAM frees on exit
→ run arms SEQUENTIALLY (single A10G; see memory gpu-generation-sequential).
R1/R2/R3 pass an absolute --adapter; grep the log for "🔌 Applying LoRA" to
confirm it actually loaded (DUMP GOTCHA). Run from the repo root (or with the
repo root on PYTHONPATH) so `tools.*` imports resolve.
"""
import argparse
import json
import time
from pathlib import Path

from tools.raft_data import render_prompt, SHORT_ANSWER_INSTRUCTION, _normalize

ROOT = str(Path(__file__).resolve().parents[1])
UC = f"{ROOT}/examples/usecase4_bedrock_userguide"
MAX_NEW_TOKENS = 48
SFT_FORMAT = "chat"


# --------------------------------------------------------------- pure core
def load_recall_map(recall: dict) -> dict:
    """question -> {retrieved_ids, hit, gold_chunk_id} from recall_at_k_top5.json."""
    recs = recall["records"] if isinstance(recall, dict) else recall
    out = {}
    for r in recs:
        out[r["question"]] = {
            "retrieved_ids": list(r["retrieved_ids"]),
            "hit": bool(r["hit"]),
            "gold_chunk_id": r.get("gold_chunk_id"),
        }
    return out


def select_items(items: list, recall_map: dict) -> list:
    """The 481 facts: drop \\bpage\\b trivia, require each to be in the recall table."""
    from tools.run_knowledge_measurement import filter_items
    kept = filter_items(items, True, None)
    for it in kept:
        if it["question"] not in recall_map:
            raise KeyError(f"item not in recall table: {it['question']!r}")
    return kept


def _answer_in_context(answer: str, context_texts: list) -> bool:
    gold = _normalize(answer or "")
    if not gold:
        return False
    ctx = _normalize("\n\n".join(context_texts))
    return gold in ctx


def build_rows(items: list, recall_map: dict, id2text: dict, *,
               arm: str, mode: str, generate) -> list:
    """Render each item's prompt, generate, and return per-fact rows.

    `generate` is injected (str->str) so this is GPU-free and unit-testable.
    """
    rows = []
    for it in items:
        q = it["question"]
        gold = it.get("answer", "")
        if mode == "rag":
            info = recall_map[q]
            rids = info["retrieved_ids"]
            ctx = [id2text[i] for i in rids]
            prompt = render_prompt(SHORT_ANSWER_INSTRUCTION, ctx, q)
            hit = info["hit"]
            aic = _answer_in_context(gold, ctx)
        elif mode == "closedbook":
            rids = []
            prompt = q
            hit = None
            aic = False
        else:
            raise ValueError(f"unknown mode {mode!r}")
        pred = generate(prompt)
        rows.append({
            "arm": arm,
            "chunk_id": it.get("chunk_id"),
            "question": q,
            "prompt": prompt,
            "prediction": pred,
            "retrieved_ids": rids,
            "hit": hit,
            "answer_in_context": aic,
            "gold": gold,
        })
    return rows


# --------------------------------------------------------------- GPU driver
def main():
    ap = argparse.ArgumentParser(description="RAFT Phase-2 RAG-mode prediction dump")
    ap.add_argument("--arm", required=True, help="arm label (R0/R1/R2/R3)")
    ap.add_argument("--mode", required=True, choices=["rag", "closedbook"])
    ap.add_argument("--adapter", required=True, help="absolute adapter dir, or the literal BASE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--config", default=f"{UC}/config_knowledge_e2b.json")
    ap.add_argument("--inventory", default=f"{UC}/eval_500.json")
    ap.add_argument("--recall", default=f"{ROOT}/results/knowledge/uc4/recall_at_k_top5.json")
    ap.add_argument("--chunks", default=f"{UC}/chunks.json")
    args = ap.parse_args()

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
        with open(args.log, "a") as f:
            f.write(line)
        print(line, end="", flush=True)

    data = json.loads(Path(args.inventory).read_text())
    raw = data["items"] if isinstance(data, dict) and "items" in data else data
    recall_map = load_recall_map(json.loads(Path(args.recall).read_text()))
    items = select_items(raw, recall_map)
    assert len(items) == 481, f"expected 481 facts, got {len(items)}"

    id2text = {}
    if args.mode == "rag":
        chunks = json.loads(Path(args.chunks).read_text())
        id2text = {int(c["chunk_id"]): c["text"] for c in chunks}

    log(f"{len(items)} facts; arm={args.arm} mode={args.mode} adapter={args.adapter}")

    from eval.knowledge_probe import build_generate_fn
    cfg = json.loads(Path(args.config).read_text())
    adapter_dir = None if args.adapter == "BASE" else args.adapter
    generate = build_generate_fn(adapter_dir, cfg, MAX_NEW_TOKENS, SFT_FORMAT)

    t0 = time.time()
    rows = build_rows(items, recall_map, id2text,
                      arm=args.arm, mode=args.mode, generate=generate)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")

    n_refusal = sum(1 for r in rows if not (r["prediction"] or "").strip())
    n_hit = sum(1 for r in rows if r["hit"] is True)
    n_aic = sum(1 for r in rows if r["answer_in_context"])
    log(f"ALL DONE arm={args.arm} n={len(rows)} hit={n_hit} answer_in_context={n_aic} "
        f"empty_pred={n_refusal} in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
