"""Orchestrator for the knowledge-measurement pipeline (docs/knowledge_measurement_plan.md).

Runs the three stores against one inventory in a single process (load each model once):
  S0  base model, closed-book          (contamination baseline)
  S1  base + LoRA adapter, closed-book  (absorbed knowledge)
  S2  vector-DB recall@k                (retrieval ceiling / mediator)
then joins them into a per-fact table + a combined profile (coverage, retention-lift
over base, recall). The pure join/alignment logic is unit-tested in
tests/test_run_knowledge_measurement.py; run() is the heavy integration path.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable


# --- pure join / alignment ----------------------------------------------

def align_fidelities(s0_facts: list[dict], s1_facts: list[dict]) -> tuple[list, list, list]:
    """Return (base_fidelities, ft_fidelities, robust_flags), aligned by chunk_id.

    robust_flags are the S1 (fine-tuned) robustness flags. Raises if the two runs do
    not describe the same facts in the same order.
    """
    if len(s0_facts) != len(s1_facts):
        raise ValueError(f"S0/S1 length mismatch: {len(s0_facts)} vs {len(s1_facts)}")
    base, ft, robust = [], [], []
    for a, b in zip(s0_facts, s1_facts):
        if a.get("chunk_id") != b.get("chunk_id"):
            raise ValueError(f"chunk_id misalignment: {a.get('chunk_id')} vs {b.get('chunk_id')}")
        base.append(a["fact_fidelity"])
        ft.append(b["fact_fidelity"])
        robust.append(b.get("robust", False))
    return base, ft, robust


def build_per_fact_table(s0_facts: list[dict], s1_facts: list[dict],
                         recall_records: list[dict]) -> list[dict]:
    """Join S0, S1 and recall into one row per fact."""
    recall_by_chunk = {r["gold_chunk_id"]: r["hit"] for r in recall_records}
    rows = []
    for a, b in zip(s0_facts, s1_facts):
        cid = b.get("chunk_id")
        rows.append({
            "fact_id": b.get("fact_id"),
            "chunk_id": cid,
            "base_fidelity": a["fact_fidelity"],
            "ft_fidelity": b["fact_fidelity"],
            "absorbed": b["fact_fidelity"] > a["fact_fidelity"],
            "recall_hit": recall_by_chunk.get(cid),
        })
    return rows


# --- helpers ------------------------------------------------------------

_PAGE_RE = re.compile(r"\bpage\b", re.IGNORECASE)


def filter_items(items: list[dict], exclude_page_trivia: bool = False,
                 limit: int | None = None) -> list[dict]:
    """Optionally drop TOC page-number trivia and/or cap the number of items."""
    out = items
    if exclude_page_trivia:
        out = [it for it in out
               if not _PAGE_RE.search(f"{it.get('question','')} {it.get('answer','')}")]
    if limit is not None:
        out = out[:limit]
    return out


def load_judge_client(creds_path: str | None):
    """Build a CodeVector (OpenAI-compatible) judge client, or None for the heuristic judge."""
    if not creds_path:
        return None
    from openai import OpenAI
    creds = json.loads(Path(creds_path).read_text())["provider"]["codevector"]["options"]
    return OpenAI(api_key=creds["apiKey"], base_url=creds["baseURL"])


def _load_items(inventory_path: str) -> list[dict]:
    data = json.loads(Path(inventory_path).read_text())
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognised inventory schema in {inventory_path}")


# --- integration run ----------------------------------------------------

def run(inventory_path: str, config_path: str, adapter_dir: str, out_path: str,
        recall_collection: str | None = None, top_k: int = 5,
        creds_path: str | None = None, judge_model: str = "deepseek-v4-lite",
        max_new_tokens: int = 32, sft_format: str = "chat", tau: float = 0.5,
        exclude_page_trivia: bool = False, limit: int | None = None) -> dict[str, Any]:
    from fastembed import TextEmbedding
    from vectorstore.registry import get_store
    from eval.knowledge_probe import build_generate_fn, score_fact
    from eval.knowledge_profile import knowledge_profile
    from eval.recall_at_k import score_inventory

    cfg = json.loads(Path(config_path).read_text())
    items = filter_items(_load_items(inventory_path), exclude_page_trivia, limit)
    judge_client = load_judge_client(creds_path)

    # S0 base
    gen0 = build_generate_fn(None, cfg, max_new_tokens, sft_format)
    s0 = [score_fact(it, gen0, judge_client, tau, judge_model=judge_model) for it in items]
    # S1 adapter
    gen1 = build_generate_fn(adapter_dir, cfg, max_new_tokens, sft_format)
    s1 = [score_fact(it, gen1, judge_client, tau, judge_model=judge_model) for it in items]
    # S2 recall@k
    collection = recall_collection or cfg["collection"]
    embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
    store = get_store(cfg)
    recall = score_inventory(items, embedder, store, collection, top_k)

    base_f, ft_f, robust = align_fidelities(s0, s1)
    profile = knowledge_profile(ft_f, base_f, robust, tau)
    table = build_per_fact_table(s0, s1, recall["records"])

    result = {
        "config": {
            "base_model": cfg.get("llm_model"), "adapter_dir": adapter_dir,
            "recall_collection": collection, "top_k": top_k,
            "judge_model": judge_model if judge_client else "heuristic",
            "max_new_tokens": max_new_tokens, "sft_format": sft_format, "tau": tau,
            "n_items": len(items), "exclude_page_trivia": exclude_page_trivia,
        },
        "profile": profile,
        "recall_summary": recall["summary"],
        "n_judge_failed": sum(f.get("n_judge_failed", 0) for f in s0 + s1),
        "per_fact": table,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    return {"profile": profile, "recall_summary": recall["summary"],
            "n_judge_failed": result["n_judge_failed"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Knowledge-measurement orchestrator (S0/S1/S2)")
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True, help="S1 adapter dir (S0 base is derived)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recall-collection", default=None, help="override collection for recall@k")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--creds", default=None, help="CodeVector creds json; omit -> heuristic judge")
    ap.add_argument("--judge-model", default="deepseek-v4-lite")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--sft-format", default="chat")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--exclude-page-trivia", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = run(args.inventory, args.config, args.adapter, args.out,
              recall_collection=args.recall_collection, top_k=args.top_k,
              creds_path=args.creds, judge_model=args.judge_model,
              max_new_tokens=args.max_new_tokens, sft_format=args.sft_format, tau=args.tau,
              exclude_page_trivia=args.exclude_page_trivia, limit=args.limit)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
