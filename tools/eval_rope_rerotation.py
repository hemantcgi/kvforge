"""E6 — RoPE re-rotation: measure attention-KL divergence with vs without K re-rotation.

Compares 5 conditions:
  A. True prefill (text-in-context — ground truth)
  B. Full-token injection, NO re-rotation (current approach)
  C. Full-token injection, WITH RoPE re-rotation (proposed fix)
  D. Mean-pool injection, NO re-rotation (current approach)
  E. Mean-pool injection, WITH approximate re-rotation (heuristic)

Reports per-layer KL divergence and cosine distance for each condition vs A.

Usage:
    python3 tools/eval_rope_rerotation.py \\
        --config examples/usecase4_bedrock_userguide/config.json \\
        --output results/rope_rerotation/uc4_results.json \\
        --max-samples 15
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import math
from pathlib import Path

import numpy as np
import torch

sys_path = str(Path(__file__).resolve().parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from eval import splits
from core.kv_utils import rerotate_keys


# ── Metrics ──────────────────────────────────────────────────────────────────


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1 - np.dot(a, b))


# ── Model / chunk helpers ──────────────────────────────────────────────────────


def _load_model(cfg: dict):
    import core.model_loader as model_loader
    import core.version as ver
    os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
    model_loader.init(cfg)
    ver.init(cfg)
    lora_ckpt = ver.load().get("checkpoint_path")
    return model_loader.reload(lora_ckpt, attn_implementation="eager")


def _retrieve_chunks(question: str, cfg: dict, max_chunks: int = 5) -> list[dict]:
    from fastembed import TextEmbedding
    from pipeline.bedrock_rag import Config, _run_search
    from vectorstore.registry import get_store

    embed_model = cfg.get("embed_model", cfg.get("addon_config", {}).get("indexing", {}).get("embed_model", "BAAI/bge-small-en-v1.5"))
    embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)
    store = get_store(cfg)
    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    flat_cfg = {**cfg, **indexing_cfg, **cfg.get("addon_config", {}).get("inference", {})}
    from pipeline.bedrock_rag import Config as RagConfig
    rag_cfg = RagConfig(**{k: flat_cfg[k] for k in RagConfig.__dataclass_fields__ if k in flat_cfg})
    hits = _run_search(question, embedder, store, rag_cfg)
    chunks = []
    for h in hits[:max_chunks]:
        payload = h.payload
        chunks.append({
            "chunk_id": h.id,
            "text": payload["text"],
            "kv_cache": payload.get("kv_cache"),
            "kv_version": payload.get("kv_version"),
        })
    return chunks


def _chunk_token_lengths(chunks: list[dict], tokenizer, max_length: int = 512) -> list[int]:
    lengths = []
    for c in chunks:
        toks = tokenizer(c["text"], truncation=True, max_length=max_length, add_special_tokens=False)
        lengths.append(len(toks["input_ids"]))
    return lengths


def _generate_one_token_attn(model, tokenizer, inputs, past_key_values=None):
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=past_key_values,
            max_new_tokens=1,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return outputs.attentions[0], outputs.sequences[0, -1].item()


def _attention_to_chunk_probs(attn_layer: torch.Tensor, chunk_lengths: list[int]) -> np.ndarray:
    attn = attn_layer.float().mean(dim=1).squeeze(0).cpu().numpy()
    key_probs = attn[-1, :]
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


# ── Load questions ──────────────────────────────────────────────────────────────


def _load_questions(cfg: dict, max_samples: int | None) -> list[dict]:
    base = Path(cfg.get("version_file", "config.json")).parent
    uc = cfg.get("use_case", "") or str(base).lower()
    try:
        if "squad" in uc.lower() or "usecase3" in str(base).lower():
            split = splits.load_squad_split(
                train_path=base / "data" / "train-v2.0.json",
                dev_path=base / "data" / "dev-v2.0.json",
                sample_dev=max_samples, auto_download=True,
            )
            return split.get("dev", [])
        if "pubmed" in uc.lower() or "usecase2" in str(base).lower():
            split = splits.load_pubmedqa_split(
                train_path=base / "data" / "train_set.json",
                test_path=base / "data" / "test_set.json",
                sample_test=max_samples, auto_download=True,
            )
            return split.get("test", [])
    except Exception:
        pass
    faq_path = base / "faqs.json"
    if faq_path.exists():
        rows = json.loads(faq_path.read_text())
        rows = [{"question": r.get("question", ""), "answer": r.get("answer", "")} for r in rows]
        if max_samples:
            rows = rows[:max_samples]
        return rows
    return []


# ── True prefill ────────────────────────────────────────────────────────────────


def _prefill_chunk_probs(question: str, chunks: list[dict], model, tokenizer, num_layers: int) -> np.ndarray:
    """Return per-layer chunk attention probs for true prefill.

    Returns: [num_layers, num_chunks] float
    """
    context = "\n\n---\n\n".join(
        f"[page {c.get('page') or i+1}]\n{c['text']}" for i, c in enumerate(chunks)
    )
    prompt = (
        "Using only the context below, answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1600,
                       add_special_tokens=False).to(model.device)
    chunk_lengths = _chunk_token_lengths(chunks, tokenizer)
    attn_tuple, _ = _generate_one_token_attn(model, tokenizer, inputs)
    probs = np.stack([_attention_to_chunk_probs(attn, chunk_lengths) for attn in attn_tuple])
    return probs


# ── KV injection with re-rotation ──────────────────────────────────────────────


def _compute_fulltoken_kv(chunks: list[dict], model, tokenizer, cfg: dict) -> list:
    """Compute per-token KV for each chunk.

    Returns list of [num_layers, 2, num_kv_heads, seq_len, head_dim] float16 arrays.
    """
    import core.kv_utils as kv_utils
    all_kvs = []
    for c in chunks:
        inputs = tokenizer(c["text"], return_tensors="pt", truncation=True, max_length=512,
                           add_special_tokens=False).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
        chunk_kv = kv_utils.compute_per_token_kv(outputs.past_key_values)
        all_kvs.append(chunk_kv)
    return all_kvs


def _build_injected_cache(
    all_kvs: list, chunk_lengths: list[int], model,
    rerotate: bool = False, inv_freq: torch.Tensor = None,
) -> tuple:
    """Build DynamicCache from per-chunk KV arrays.

    If ``rerotate=True``, applies RoPE rotation delta to K vectors so they
    are correct for their new concatenated positions instead of their
    original in-chunk positions.

    For full-token mode: concatenates per-chunk sequences along seq dim.
    Each chunk's tokens move from in-chunk positions 0..L-1 to
    global positions offset..offset+L-1 where offset = sum of previous lengths.
    """
    num_layers, _, num_kv_heads, _, head_dim = all_kvs[0].shape
    chunk_lengths = [int(l) for l in chunk_lengths]

    offsets = [0]
    for l in chunk_lengths:
        offsets.append(offsets[-1] + l)

    per_layer = []
    for layer_idx in range(num_layers):
        ks, vs = [], []
        for chunk_idx, chunk_kv in enumerate(all_kvs):
            # chunk_kv[layer_idx]: [2, num_kv_heads, seq_len, head_dim] numpy
            layer = torch.from_numpy(chunk_kv[layer_idx].astype(np.float16))
            k = layer[0]  # [num_kv_heads, seq_len, head_dim]
            v = layer[1]

            if rerotate and inv_freq is not None:
                seq_len = k.shape[1]
                old_pos = torch.arange(seq_len, dtype=torch.float)
                new_pos = old_pos + offsets[chunk_idx]
                delta = new_pos - old_pos  # = offsets[chunk_idx] for all tokens
                k = rerotate_keys(k.unsqueeze(0), inv_freq, delta).squeeze(0)

            ks.append(k)
            vs.append(v)

        target_dtype = next(model.parameters()).dtype
        k = torch.cat(ks, dim=1).unsqueeze(0).to(device=model.device, dtype=target_dtype)
        v = torch.cat(vs, dim=1).unsqueeze(0).to(device=model.device, dtype=target_dtype)
        per_layer.append((k, v))

    try:
        from transformers.cache_utils import DynamicCache
        return DynamicCache(ddp_cache_data=per_layer)
    except (ImportError, TypeError):
        return tuple(per_layer)


def _build_meanpool_cache(
    chunks: list[dict], cfg: dict, model,
    rerotate: bool = False, inv_freq: torch.Tensor = None,
    chunk_lengths: list[int] = None,
) -> tuple:
    """Build DynamicCache from mean-pooled KV.

    If ``rerotate=True``, approximate: treat mean-pooled K as if it were at
    position chunk_len/2 and re-rotate to the chunk's index position.
    """
    import core.kv_utils as kv_utils
    import core.model_loader as model_loader
    num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)
    kv_shape = (num_layers, 2, num_kv_heads, head_dim)

    chunk_kvs = []
    for idx, c in enumerate(chunks):
        arr = kv_utils.deserialize_kv(c["kv_cache"], shape=kv_shape)
        if rerotate and inv_freq is not None and chunk_lengths is not None:
            # Approximate: mean-pooled K is at old_pos = chunk_lengths[idx] / 2
            old_pos = chunk_lengths[idx] / 2.0
            new_pos = float(idx)  # each chunk = 1 virtual position
            delta = torch.tensor([new_pos - old_pos], dtype=torch.float)
            for layer_idx in range(num_layers):
                k = torch.from_numpy(arr[layer_idx, 0].astype(np.float16))
                k = rerotate_keys(k.unsqueeze(0).unsqueeze(1), inv_freq, delta).squeeze()
                arr[layer_idx, 0] = k.numpy().astype(np.float16)
        chunk_kvs.append(arr)

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


def _kv_chunk_probs(
    question: str, chunks: list[dict], model, tokenizer, cfg: dict,
    mode: str, inv_freq: torch.Tensor = None, all_kvs: list = None,
    chunk_lengths: list[int] = None,
) -> np.ndarray:
    """Return per-layer chunk attention probs for a KV injection mode.

    mode: "fulltoken", "fulltoken_rerotated", "meanpool", "meanpool_rerotated"
    """
    rerotate = "_rerotated" in mode
    base_mode = mode.replace("_rerotated", "")

    if base_mode == "meanpool":
        past_kv = _build_meanpool_cache(chunks, cfg, model, rerotate=rerotate,
                                         inv_freq=inv_freq, chunk_lengths=chunk_lengths)
        cl = [1] * len(chunks)
    else:
        if all_kvs is None:
            all_kvs = _compute_fulltoken_kv(chunks, model, tokenizer, cfg)
            chunk_lengths = _chunk_token_lengths(chunks, tokenizer)
        past_kv = _build_injected_cache(all_kvs, chunk_lengths, model,
                                        rerotate=rerotate, inv_freq=inv_freq)
        cl = chunk_lengths

    prompt = f"Based on the context provided, answer: {question}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Prepend dummy tokens for position offset
    prefix_len = past_kv[0][0].shape[2] if isinstance(past_kv, tuple) else past_kv.get_seq_length()
    if prefix_len:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        prefix = torch.full((1, prefix_len), pad_id, dtype=torch.long, device=model.device)
        inputs["input_ids"] = torch.cat([prefix, inputs["input_ids"]], dim=1)
        inputs["attention_mask"] = torch.cat([torch.ones_like(prefix), inputs["attention_mask"]], dim=1)

    attn_tuple, _ = _generate_one_token_attn(model, tokenizer, inputs, past_key_values=past_kv)
    probs = np.stack([_attention_to_chunk_probs(attn, cl) for attn in attn_tuple])
    return probs


# ── Main experiment ────────────────────────────────────────────────────────────


def _evaluate(cfg: dict, questions: list[dict], max_samples: int | None) -> dict:
    import core.model_loader as model_loader
    import core.version as ver

    if max_samples:
        questions = questions[:max_samples]

    model, tokenizer = _load_model(cfg)
    ver.init(cfg)

    # Extract RoPE frequencies — search model modules recursively
    inv_freq = None
    for name, mod in model.named_modules():
        if hasattr(mod, "inv_freq"):
            inv_freq = mod.inv_freq
            print(f"Extracted inv_freq from {name}: shape={inv_freq.shape}, dtype={inv_freq.dtype}", flush=True)
            print(f"  inv_freq[:5]: {inv_freq[:5].tolist()}", flush=True)
            break
    if inv_freq is None:
        raise ValueError("Could not find rotary_emb.inv_freq in model")

    num_layers = model_loader.get_kv_shape(cfg)[0]
    layer_records = []
    for _ in range(num_layers):
        layer_records.append({
            "kl_fulltoken": [], "kl_fulltoken_rerotated": [],
            "kl_meanpool": [], "kl_meanpool_rerotated": [],
            "cos_fulltoken": [], "cos_fulltoken_rerotated": [],
            "cos_meanpool": [], "cos_meanpool_rerotated": [],
        })

    per_question = []
    for q_idx, item in enumerate(questions):
        question = item["question"]
        print(f"\n▶ Q {q_idx + 1}/{len(questions)}: {question[:80]}...", flush=True)
        chunks = _retrieve_chunks(question, cfg)
        if not chunks:
            print("   no chunks retrieved")
            continue

        try:
            # A: True prefill
            prefill_probs = _prefill_chunk_probs(question, chunks, model, tokenizer, num_layers)
            # Precompute full-token KV once for conditions B and C
            all_kvs = _compute_fulltoken_kv(chunks, model, tokenizer, cfg)
            chunk_lengths = _chunk_token_lengths(chunks, tokenizer)

            # B: Full-token no re-rotation
            print("   B: full-token (no re-rotation)...", flush=True)
            fulltoken_probs = _kv_chunk_probs(question, chunks, model, tokenizer, cfg,
                                              "fulltoken", all_kvs=all_kvs, chunk_lengths=chunk_lengths)
            # C: Full-token WITH re-rotation
            print(f"   C: full-token (WITH re-rotation), offsets={chunk_lengths}, chunks={len(all_kvs)}", flush=True)
            for ci, (ckv, clen) in enumerate(zip(all_kvs, chunk_lengths)):
                print(f"     chunk {ci}: seq_len={ckv.shape[3]}, token_lens={clen}", flush=True)
            fulltoken_rerot_probs = _kv_chunk_probs(question, chunks, model, tokenizer, cfg,
                                                     "fulltoken_rerotated", inv_freq=inv_freq,
                                                     all_kvs=all_kvs, chunk_lengths=chunk_lengths)
            # D: Mean-pool no re-rotation
            print("   D: mean-pool (no re-rotation)...", flush=True)
            meanpool_probs = _kv_chunk_probs(question, chunks, model, tokenizer, cfg,
                                             "meanpool", chunk_lengths=chunk_lengths)
            # E: Mean-pool WITH re-rotation
            print("   E: mean-pool (WITH re-rotation)...", flush=True)
            meanpool_rerot_probs = _kv_chunk_probs(question, chunks, model, tokenizer, cfg,
                                                    "meanpool_rerotated", inv_freq=inv_freq,
                                                    chunk_lengths=chunk_lengths)
        except Exception as exc:
            print(f"   error: {exc}")
            import traceback
            traceback.print_exc()
            continue

        q_record = {"question": question, "num_chunks": len(chunks), "layers": []}
        for layer in range(num_layers):
            p_pre = prefill_probs[layer]
            for label, p in [("fulltoken", fulltoken_probs[layer]),
                             ("fulltoken_rerotated", fulltoken_rerot_probs[layer]),
                             ("meanpool", meanpool_probs[layer]),
                             ("meanpool_rerotated", meanpool_rerot_probs[layer])]:
                kl = _kl_divergence(p_pre, p)
                cos = _cosine_dist(p_pre, p)
                layer_records[layer][f"kl_{label}"].append(kl)
                layer_records[layer][f"cos_{label}"].append(cos)

            q_record["layers"].append({
                "layer": layer,
                "kl_fulltoken": round(float(np.mean(layer_records[layer]["kl_fulltoken"])), 4),
                "kl_fulltoken_rerotated": round(float(np.mean(layer_records[layer]["kl_fulltoken_rerotated"])), 4),
                "kl_meanpool": round(float(np.mean(layer_records[layer]["kl_meanpool"])), 4),
                "kl_meanpool_rerotated": round(float(np.mean(layer_records[layer]["kl_meanpool_rerotated"])), 4),
            })
        per_question.append(q_record)

    per_layer = []
    for layer in range(num_layers):
        per_layer.append({
            "layer": layer,
            "kl_fulltoken": round(float(np.mean(layer_records[layer]["kl_fulltoken"])), 4),
            "kl_fulltoken_rerotated": round(float(np.mean(layer_records[layer]["kl_fulltoken_rerotated"])), 4),
            "kl_meanpool": round(float(np.mean(layer_records[layer]["kl_meanpool"])), 4),
            "kl_meanpool_rerotated": round(float(np.mean(layer_records[layer]["kl_meanpool_rerotated"])), 4),
        })

    return {
        "n_questions": len(questions),
        "per_question": per_question,
        "per_layer": per_layer,
        "mean_kl_fulltoken": round(float(np.mean([l["kl_fulltoken"] for l in per_layer])), 4),
        "mean_kl_fulltoken_rerotated": round(float(np.mean([l["kl_fulltoken_rerotated"] for l in per_layer])), 4),
        "mean_kl_meanpool": round(float(np.mean([l["kl_meanpool"] for l in per_layer])), 4),
        "mean_kl_meanpool_rerotated": round(float(np.mean([l["kl_meanpool_rerotated"] for l in per_layer])), 4),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="E6: RoPE re-rotation attention divergence")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=15)
    parser.add_argument("--eval-set", default=None, help="Direct path to eval JSON (bypasses _load_questions)")
    args = parser.parse_args()

    raw_cfg = json.loads(Path(args.config).read_text())
    addon_config = raw_cfg.get("addon_config", {})
    cfg = {**raw_cfg}
    for section in ("indexing", "inference", "training", "background", "sync", "monitoring"):
        cfg.update(addon_config.get(section, {}))

    if args.eval_set:
        eval_data = json.loads(Path(args.eval_set).read_text())
        questions = eval_data["items"] if "items" in eval_data else eval_data
    else:
        questions = _load_questions(cfg, max_samples=args.max_samples)
    if not questions:
        raise ValueError("No questions found")

    result = _evaluate(cfg, questions, max_samples=args.max_samples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n✓ Wrote {args.output}")
    print(f"  mean KL fulltoken:          {result['mean_kl_fulltoken']:.4f}")
    print(f"  mean KL fulltoken_rerotated: {result['mean_kl_fulltoken_rerotated']:.4f}")
    print(f"  mean KL meanpool:           {result['mean_kl_meanpool']:.4f}")
    print(f"  mean KL meanpool_rerotated:  {result['mean_kl_meanpool_rerotated']:.4f}")
    print(f"  KL improvement (fulltoken):  {(result['mean_kl_fulltoken'] - result['mean_kl_fulltoken_rerotated']):.4f}")
    print(f"  KL improvement (meanpool):   {(result['mean_kl_meanpool'] - result['mean_kl_meanpool_rerotated']):.4f}")


if __name__ == "__main__":
    main()
