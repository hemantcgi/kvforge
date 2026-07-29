"""E5 — Attention divergence between true prefill, full-token injected KV, and mean-pool.

Usage:

    python -m pipeline.eval_attention_divergence \
        --config examples/usecase4_bedrock_userguide/config.json \
        --output examples/usecase4_bedrock_userguide/eval_attention_divergence.json \
        --max-samples 30

The script computes per-layer attention-score distributions for a held-out
query under three conditions:

1. True prefill (text-in-context)
2. Full-token injected KV (Enhanced Tier)
3. Mean-pooled injected KV (Active Tier)

It reports KL divergence and cosine distance between the chunk-level attention
score distributions.  The real implementation requires a GPU and loads the
model with ``attn_implementation="eager"`` so that ``output_attentions=True``
works.  A deterministic ``--dry-run`` mode is kept for CI.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys_path = str(Path(__file__).resolve().parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from eval import splits


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(p || q) with smoothing."""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1 - np.dot(a, b))


def _dry_run_divergence(num_layers: int, seed: int) -> dict:
    """Generate realistic synthetic divergence curves."""
    rng = random.Random(seed)
    layers = []
    for layer in range(num_layers):
        base_mean = 0.05 + 0.25 * (1 - np.exp(-layer / 8))
        full = base_mean + rng.gauss(0, 0.02)
        mean = base_mean * 1.5 + rng.gauss(0, 0.03)
        layers.append({
            "layer": layer,
            "kl_fulltoken_vs_prefill": round(max(0.0, full), 4),
            "kl_meanpool_vs_prefill": round(max(0.0, mean), 4),
            "cosine_fulltoken_vs_prefill": round(max(0.0, full * 0.6), 4),
            "cosine_meanpool_vs_prefill": round(max(0.0, mean * 0.6), 4),
        })
    return {
        "per_layer": layers,
        "mean_kl_fulltoken": round(np.mean([l["kl_fulltoken_vs_prefill"] for l in layers]), 4),
        "mean_kl_meanpool": round(np.mean([l["kl_meanpool_vs_prefill"] for l in layers]), 4),
    }


def _load_questions(cfg: dict, max_samples: int | None) -> list[dict]:
    """Load evaluation questions from the split or fallback FAQ file."""
    base = Path(cfg.get("version_file", "config.json")).parent

    uc = cfg.get("use_case", "")
    if not uc:
        uc = str(base).lower()

    try:
        if "squad" in uc.lower() or "usecase3" in str(base).lower():
            train_path = base / "data" / "train-v2.0.json"
            dev_path = base / "data" / "dev-v2.0.json"
            faq_train = base / "faqs_train.json"
            if not faq_train.exists():
                faq_train = base / "faqs.json"
            split = splits.load_squad_split(
                train_path=train_path, dev_path=dev_path,
                faqs_train_path=faq_train if faq_train.exists() else None,
                sample_dev=max_samples, auto_download=True,
            )
            return split.get("dev", [])

        if "pubmed" in uc.lower() or "usecase2" in str(base).lower():
            train_path = base / "data" / "train_set.json"
            test_path = base / "data" / "test_set.json"
            faq_train = base / "faqs_train.json"
            if not faq_train.exists():
                faq_train = base / "faqs.json"
            split = splits.load_pubmedqa_split(
                train_path=train_path, test_path=test_path,
                faqs_train_path=faq_train if faq_train.exists() else None,
                sample_test=max_samples, auto_download=True,
            )
            return split.get("test", [])
    except Exception:
        pass

    faq_path = base / "faqs.json"
    if faq_path.exists():
        rows = json.loads(faq_path.read_text())
        rows = [{"question": r.get("question", ""), "answer": r.get("answer", "")}
                for r in rows]
        if max_samples:
            rows = rows[:max_samples]
        return rows
    return []


# ── Real GPU implementation ───────────────────────────────────────────────────


def _load_model(cfg: dict, checkpoint: str | None = None):
    """Load model with eager attention so output_attentions works."""
    import core.model_loader as model_loader
    import core.version as ver
    os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
    model_loader.init(cfg)
    ver.init(cfg)
    lora_ckpt = checkpoint or ver.load().get("checkpoint_path")
    return model_loader.reload(lora_ckpt, attn_implementation="eager")


def _retrieve_chunks(question: str, cfg: dict) -> list[dict]:
    """Retrieve top-k chunks for a question."""
    from fastembed import TextEmbedding
    from pipeline.bedrock_rag import Config, _run_search
    from vectorstore.registry import get_store

    embed_model = cfg.get("embed_model", cfg.get("addon_config", {}).get("indexing", {}).get("embed_model", "BAAI/bge-small-en-v1.5"))
    embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)
    store = get_store(cfg)

    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    flat_cfg = {**cfg, **indexing_cfg, **cfg.get("addon_config", {}).get("inference", {})}
    rag_cfg = Config(**{k: flat_cfg[k] for k in Config.__dataclass_fields__ if k in flat_cfg})

    hits = _run_search(question, embedder, store, rag_cfg)
    chunks = []
    for h in hits:
        payload = h.payload
        chunks.append({
            "chunk_id": h.id,
            "text": payload["text"],
            "page": payload.get("page"),
            "score": round(h.score, 4),
            "kv_cache": payload.get("kv_cache"),
            "kv_version": payload.get("kv_version"),
            "kv_token_path": payload.get("kv_token_path"),
            "status": payload.get("status", "active"),
        })
    return chunks


def _chunk_token_lengths(chunks: list[dict], tokenizer, max_length: int = 512) -> list[int]:
    """Return per-chunk token counts (no separators)."""
    lengths = []
    for c in chunks:
        toks = tokenizer(c["text"], truncation=True, max_length=max_length, add_special_tokens=False)
        lengths.append(len(toks["input_ids"]))
    return lengths


def _generate_one_token_attn(model, tokenizer, inputs, past_key_values=None):
    """Generate one token, return attention tuple and generated token id."""
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=past_key_values,
            max_new_tokens=1,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    # outputs.attentions is a tuple of length num_generated_tokens.
    # Each element is a tuple of layers; each layer tensor [batch, heads, q_len, k_len].
    return outputs.attentions[0], outputs.sequences[0, -1].item()


def _attention_to_chunk_probs(attn_layer: torch.Tensor, chunk_lengths: list[int]) -> np.ndarray:
    """Average over heads, take generated-token row, and sum per chunk.

    Args:
        attn_layer: [batch, heads, q_len, k_len]
        chunk_lengths: per-chunk token counts (sum <= k_len)

    Returns:
        Normalized chunk-level probability vector of length len(chunk_lengths).
    """
    attn = attn_layer.float().mean(dim=1).squeeze(0).cpu().numpy()  # [q_len, k_len]
    # Last query position corresponds to the generated token.
    key_probs = attn[-1, :]  # [k_len]
    # We only care about the chunk positions at the start of the key dimension.
    total_chunk_len = sum(chunk_lengths)
    key_probs = key_probs[:total_chunk_len]
    probs = []
    idx = 0
    for length in chunk_lengths:
        probs.append(float(key_probs[idx:idx + length].sum()))
        idx += length
    probs = np.asarray(probs)
    probs = np.clip(probs, 1e-10, 1.0)
    probs = probs / probs.sum()
    return probs


def _prefill_chunk_probs(question: str, chunks: list[dict], model, tokenizer) -> tuple[np.ndarray, np.ndarray]:
    """Return per-layer chunk attention probs for true prefill.

    Returns a tuple (probs_per_layer, raw_logits_or_empty) where probs_per_layer
    is a numpy array of shape [num_layers, num_chunks].
    """
    context = "\n\n---\n\n".join(
        f"[page {c.get('page') or i+1}]\n{c['text']}" for i, c in enumerate(chunks)
    )
    prompt = (
        "Using only the context below, answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1600).to(model.device)
    chunk_lengths = _chunk_token_lengths(chunks, tokenizer)
    attn_tuple, _ = _generate_one_token_attn(model, tokenizer, inputs)
    probs = np.stack([_attention_to_chunk_probs(attn, chunk_lengths) for attn in attn_tuple])
    return probs


def _meanpool_kv(chunks: list[dict], cfg: dict, model) -> tuple:
    """Build mean-pool past_key_values from chunk payloads."""
    import core.kv_utils as kv_utils
    import core.model_loader as model_loader
    num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)
    kv_shape = (num_layers, 2, num_kv_heads, head_dim)
    chunk_kvs = [kv_utils.deserialize_kv(c["kv_cache"], shape=kv_shape) for c in chunks]
    past_kv = kv_utils.stack_past_key_values(
        chunk_kvs, num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            for layer in past_kv.layers:
                layer.keys = layer.keys.to(model.device)
                layer.values = layer.values.to(model.device)
        else:
            past_kv = tuple((k.to(model.device), v.to(model.device)) for k, v in past_kv)
    except ImportError:
        past_kv = tuple((k.to(model.device), v.to(model.device)) for k, v in past_kv)
    return past_kv


def _fulltoken_kv(chunks: list[dict], model, tokenizer, cfg: dict) -> tuple:
    """Build full-token past_key_values by recomputing per-chunk KV."""
    import core.kv_utils as kv_utils
    import core.model_loader as model_loader
    num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)

    all_kvs = []
    for c in chunks:
        inputs = tokenizer(c["text"], return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
        chunk_kv = kv_utils.compute_per_token_kv(outputs.past_key_values)
        all_kvs.append(chunk_kv)

    # chunk_kv shape: [num_layers, 2, num_kv_heads, seq_len, head_dim]
    per_layer = []
    for layer_idx in range(num_layers):
        ks, vs = [], []
        for chunk_kv in all_kvs:
            layer = torch.from_numpy(chunk_kv[layer_idx].astype(np.float16))
            ks.append(layer[0])
            vs.append(layer[1])
        k = torch.cat(ks, dim=1).unsqueeze(0).to(model.device)
        v = torch.cat(vs, dim=1).unsqueeze(0).to(model.device)
        per_layer.append((k, v))
    try:
        from transformers.cache_utils import DynamicCache
        past_kv = DynamicCache(ddp_cache_data=per_layer)
    except (ImportError, TypeError):
        past_kv = tuple(per_layer)
    return past_kv


def _kv_chunk_probs(question: str, chunks: list[dict], model, tokenizer, cfg: dict, mode: str) -> np.ndarray:
    """Return per-layer chunk attention probs for a KV injection mode.

    mode: "meanpool" or "fulltoken".
    """
    if mode == "meanpool":
        past_kv = _meanpool_kv(chunks, cfg, model)
        chunk_lengths = [1] * len(chunks)  # mean-pool: one position per chunk
    else:
        past_kv = _fulltoken_kv(chunks, model, tokenizer, cfg)
        chunk_lengths = _chunk_token_lengths(chunks, tokenizer)

    prompt = f"Based on the context provided, answer: {question}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prefix_len = past_kv[0][0].shape[2] if isinstance(past_kv, tuple) else past_kv.get_seq_length()
    if prefix_len:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        prefix = torch.full((1, prefix_len), pad_id, dtype=torch.long, device=model.device)
        inputs["input_ids"] = torch.cat([prefix, inputs["input_ids"]], dim=1)
        inputs["attention_mask"] = torch.cat([torch.ones_like(prefix), inputs["attention_mask"]], dim=1)

    attn_tuple, _ = _generate_one_token_attn(model, tokenizer, inputs, past_key_values=past_kv)
    probs = np.stack([_attention_to_chunk_probs(attn, chunk_lengths) for attn in attn_tuple])
    return probs


def _evaluate_real(cfg: dict, questions: list[dict], max_samples: int | None, checkpoint: str | None) -> dict:
    import core.model_loader as model_loader
    import core.version as ver

    if max_samples:
        questions = questions[:max_samples]

    model, tokenizer = _load_model(cfg, checkpoint=checkpoint)
    ver.init(cfg)

    num_layers = model_loader.get_kv_shape(cfg)[0]
    layer_records = []
    for _ in range(num_layers):
        layer_records.append({
            "kl_fulltoken_vs_prefill": [],
            "kl_meanpool_vs_prefill": [],
            "cosine_fulltoken_vs_prefill": [],
            "cosine_meanpool_vs_prefill": [],
        })

    per_question = []
    for q_idx, item in enumerate(questions):
        question = item["question"]
        print(f"▶ Question {q_idx + 1}/{len(questions)}: {question[:80]}…", flush=True)
        chunks = _retrieve_chunks(question, cfg)
        if not chunks:
            print("   ⚠️ no chunks retrieved")
            continue
        try:
            prefill_probs = _prefill_chunk_probs(question, chunks, model, tokenizer)
            meanpool_probs = _kv_chunk_probs(question, chunks, model, tokenizer, cfg, "meanpool")
            fulltoken_probs = _kv_chunk_probs(question, chunks, model, tokenizer, cfg, "fulltoken")
        except Exception as exc:
            print(f"   ⚠️ attention error: {exc}")
            continue

        q_record = {"question": question, "num_chunks": len(chunks), "layers": []}
        for layer in range(num_layers):
            p_pre = prefill_probs[layer]
            p_mean = meanpool_probs[layer]
            p_full = fulltoken_probs[layer]
            kl_full = _kl_divergence(p_pre, p_full)
            kl_mean = _kl_divergence(p_pre, p_mean)
            cos_full = _cosine_dist(p_pre, p_full)
            cos_mean = _cosine_dist(p_pre, p_mean)
            layer_records[layer]["kl_fulltoken_vs_prefill"].append(kl_full)
            layer_records[layer]["kl_meanpool_vs_prefill"].append(kl_mean)
            layer_records[layer]["cosine_fulltoken_vs_prefill"].append(cos_full)
            layer_records[layer]["cosine_meanpool_vs_prefill"].append(cos_mean)
            q_record["layers"].append({
                "layer": layer,
                "kl_fulltoken_vs_prefill": round(kl_full, 4),
                "kl_meanpool_vs_prefill": round(kl_mean, 4),
                "cosine_fulltoken_vs_prefill": round(cos_full, 4),
                "cosine_meanpool_vs_prefill": round(cos_mean, 4),
            })
        per_question.append(q_record)

    per_layer = []
    for layer in range(num_layers):
        per_layer.append({
            "layer": layer,
            "kl_fulltoken_vs_prefill": round(float(np.mean(layer_records[layer]["kl_fulltoken_vs_prefill"])), 4),
            "kl_meanpool_vs_prefill": round(float(np.mean(layer_records[layer]["kl_meanpool_vs_prefill"])), 4),
            "cosine_fulltoken_vs_prefill": round(float(np.mean(layer_records[layer]["cosine_fulltoken_vs_prefill"])), 4),
            "cosine_meanpool_vs_prefill": round(float(np.mean(layer_records[layer]["cosine_meanpool_vs_prefill"])), 4),
        })

    return {
        "n_questions": len(questions),
        "per_question": per_question,
        "per_layer": per_layer,
        "mean_kl_fulltoken": round(float(np.mean([l["kl_fulltoken_vs_prefill"] for l in per_layer])), 4),
        "mean_kl_meanpool": round(float(np.mean([l["kl_meanpool_vs_prefill"] for l in per_layer])), 4),
        "mean_cosine_fulltoken": round(float(np.mean([l["cosine_fulltoken_vs_prefill"] for l in per_layer])), 4),
        "mean_cosine_meanpool": round(float(np.mean([l["cosine_meanpool_vs_prefill"] for l in per_layer])), 4),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="E5: Attention divergence")
    parser.add_argument("--config", required=True, help="Path to use-case config.json")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Deterministic simulation")
    parser.add_argument("--dry-run-seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None,
                        help="Override LoRA checkpoint path")
    args = parser.parse_args()

    raw_cfg = json.loads(Path(args.config).read_text())
    addon_config = raw_cfg.get("addon_config", {})
    cfg = {**raw_cfg}
    for section in ("indexing", "inference", "training", "background", "sync", "monitoring"):
        cfg.update(addon_config.get(section, {}))

    num_layers = cfg.get("kv_num_layers", 28)
    questions = _load_questions(cfg, max_samples=args.max_samples)
    if not questions:
        raise ValueError("No questions found")

    if args.dry_run:
        result = _dry_run_divergence(num_layers, seed=args.dry_run_seed)
    else:
        result = _evaluate_real(cfg, questions, max_samples=args.max_samples, checkpoint=args.checkpoint)

    result["n_questions"] = len(questions)
    result["dry_run"] = args.dry_run

    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"✓ Wrote {args.output}")
    print(f"  Mean KL full-token={result.get('mean_kl_fulltoken', 0):.4f}  "
          f"mean-pool={result.get('mean_kl_meanpool', 0):.4f}")


if __name__ == "__main__":
    main()
