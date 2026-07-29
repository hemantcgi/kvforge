"""Extend phase-quality evaluation to record per-chunk factual accuracy for KV modes.

For each evaluation question that has source_chunk_ids in the FAQ, force
text_rag, kv_meanpool, and kv_fulltoken, score the prediction, and join the
result with the chunk's KDS score from the vector store.

Output: JSON with per_chunk and per_question tables.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Ensure project root is importable for eval/, pipeline/, core/, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import metrics
from pipeline.kv_inference import answer_with_mode
from core import model_loader, version as ver
from vectorstore.registry import get_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--faqs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--judge-model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    addon = cfg.get("addon_config", {})
    for section in ("indexing", "inference", "training", "background", "sync", "monitoring"):
        cfg.update(addon.get(section, {}))

    ver.init(cfg)
    model_loader.init(cfg)
    version = ver.load()
    cfg["checkpoint_path"] = version.get("checkpoint_path")
    cfg["phase"] = version.get("phase", 1)
    cfg["current_lora_version"] = version.get("current_lora_version", 0)

    store = get_store(cfg)
    faqs = json.loads(Path(args.faqs).read_text())
    if args.max_samples:
        faqs = faqs[:args.max_samples]

    model, tokenizer = model_loader.load(cfg.get("checkpoint_path"))
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                       max_new_tokens=256, do_sample=False)

    per_question = []
    per_chunk = {}

    for idx, faq in enumerate(faqs):
        q = faq.get("question", "")
        gt = faq.get("answer", "")
        chunk_ids = faq.get("source_chunk_ids", [])
        if not q or not gt or not chunk_ids:
            continue
        print(f"[{idx+1}/{len(faqs)}] {q[:60]}...")
        for mode in ("text_rag", "kv_meanpool", "kv_fulltoken"):
            try:
                pred, used = answer_with_mode(q, cfg, force_mode=mode)
            except Exception as e:
                print(f"  mode={mode} error: {e}")
                continue
            em = metrics.exact_match(pred, gt)
            f1 = metrics.token_f1(pred, gt)
            judge = metrics.llm_judge(q, pred, gt)
            factual_acc = 0.5 * f1 + 0.5 * float(judge["factually_correct"])
            per_question.append({
                "question": q,
                "mode": mode,
                "source_chunk_ids": chunk_ids,
                "ground_truth": gt,
                "prediction": pred,
                "em": em,
                "token_f1": f1,
                "judge_correct": int(judge["factually_correct"]),
                "factual_acc": factual_acc,
            })
            for cid in chunk_ids:
                if cid not in per_chunk:
                    per_chunk[cid] = {"text_rag": [], "kv_meanpool": [], "kv_fulltoken": []}
                per_chunk[cid][mode].append(factual_acc)

    # Fetch KDS for each chunk
    chunk_rows = []
    for cid, mode_scores in per_chunk.items():
        try:
            results, _ = store.scroll(cfg["collection"], limit=10000, with_payload=True, with_vectors=False)
            payload = None
            for r in results:
                if str(r.id) == str(cid):
                    payload = r.payload
                    break
            kds = payload.get("kds") if payload else None
            row = {"chunk_id": cid, "kds": kds}
            for mode in ("text_rag", "kv_meanpool", "kv_fulltoken"):
                scores = mode_scores[mode]
                row[f"{mode}_mean_factual_acc"] = sum(scores) / len(scores) if scores else None
                row[f"{mode}_n"] = len(scores)
            chunk_rows.append(row)
        except Exception as e:
            print(f"Error fetching chunk {cid}: {e}")

    output = {
        "config": args.config,
        "per_question": per_question,
        "per_chunk": chunk_rows,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
