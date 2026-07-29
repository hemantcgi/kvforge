"""Controlled RoPE re-rotation experiments — isolate position mismatch from prompt confounds.

Experiments:
1. Multi-chunk: same text, compare true concat vs inject with/without re-rotation
2. Per-token attention (not chunk-level)
3. Single-chunk position shift test
4. Statistical power: confidence intervals over 30+ questions
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import splits, metrics as eval_metrics
import core.kv_utils as kv_utils


# ── Metrics ──────────────────────────────────────────────────────────────────
def _kl(p, q):
    p = np.clip(p, 1e-10, 1.0); q = np.clip(q, 1e-10, 1.0)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

def _cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return float(1 - np.dot(a, b))


# ── Model / Data ─────────────────────────────────────────────────────────────
def load_model(cfg: dict):
    import core.model_loader as ml
    import core.version as ver
    os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
    ml.init(cfg); ver.init(cfg)
    return ml.reload(None, attn_implementation="eager")

def retrieve_chunks(question: str, cfg: dict, max_chunks: int = 3):
    from fastembed import TextEmbedding
    from pipeline.bedrock_rag import Config, _run_search
    from vectorstore.registry import get_store
    em = TextEmbedding(
        cfg.get("embed_model", cfg.get("addon_config",{}).get("indexing",{}).get("embed_model","BAAI/bge-small-en-v1.5")),
        show_download_progress=False)
    store = get_store(cfg)
    ic = cfg.get("addon_config",{}).get("indexing",{})
    fc = {**cfg, **ic, **cfg.get("addon_config",{}).get("inference",{})}
    rc = Config(**{k: fc[k] for k in Config.__dataclass_fields__ if k in fc})
    hits = _run_search(question, em, store, rc)
    return [h.payload["text"] for h in hits[:max_chunks]]


# ── KV computation ───────────────────────────────────────────────────────────
def compute_chunk_kv(text: str, model, tokenizer, max_len=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    return kv_utils.compute_per_token_kv(outputs.past_key_values), inputs["input_ids"].shape[1]


def get_inv_freq(model):
    for _, mod in model.named_modules():
        if hasattr(mod, "inv_freq"):
            return mod.inv_freq
    raise ValueError("inv_freq not found")


def rerotate_kv(chunk_kv, layer_idx, inv_freq, delta):
    """Re-rotate K vectors in a chunk by delta positions."""
    layer = torch.from_numpy(chunk_kv[layer_idx].astype(np.float16))
    k, v = layer[0], layer[1]  # [num_kv_heads, seq_len, head_dim]
    delta_t = torch.full((k.shape[1],), float(delta), dtype=torch.float)
    k_new = kv_utils.rerotate_keys(k.unsqueeze(0), inv_freq, delta_t).squeeze(0)
    return k_new, v


def build_dynamic_cache(all_kvs, chunk_lengths, model, inv_freq=None):
    """Build DynamicCache from per-chunk KVs, optionally with re-rotation."""
    num_layers = all_kvs[0].shape[0]
    target_dtype = next(model.parameters()).dtype
    offsets = [0]
    for l in chunk_lengths:
        offsets.append(offsets[-1] + l)

    per_layer = []
    for layer_idx in range(num_layers):
        ks, vs = [], []
        for ci, (ckv, clen) in enumerate(zip(all_kvs, chunk_lengths)):
            if inv_freq is not None:
                k, v = rerotate_kv(ckv, layer_idx, inv_freq, offsets[ci])
            else:
                layer = torch.from_numpy(ckv[layer_idx].astype(np.float16))
                k, v = layer[0], layer[1]
            ks.append(k); vs.append(v)
        k = torch.cat(ks, dim=1).unsqueeze(0).to(device=model.device, dtype=target_dtype)
        v = torch.cat(vs, dim=1).unsqueeze(0).to(device=model.device, dtype=target_dtype)
        per_layer.append((k, v))
    from transformers.cache_utils import DynamicCache
    return DynamicCache(ddp_cache_data=per_layer)


def get_one_token_attn(model, tokenizer, prompt_text, cache=None, max_len=1600):
    """Generate one token, return attention tuple and token id."""
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
    if cache is not None:
        prefix_len = cache.get_seq_length()
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        prefix = torch.full((1, prefix_len), pad_id, dtype=torch.long, device=model.device)
        inputs["input_ids"] = torch.cat([prefix, inputs["input_ids"]], dim=1)
        inputs["attention_mask"] = torch.cat([torch.ones_like(prefix), inputs["attention_mask"]], dim=1)
    with torch.no_grad():
        outputs = model.generate(**inputs, past_key_values=cache, max_new_tokens=1,
            do_sample=False, output_attentions=True, return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)
    return outputs.attentions[0], outputs.sequences[0, -1].item()


def attn_to_per_token(attn_tuple, total_seq_len):
    """Return per-layer, per-key-token attention vectors as np arrays.
    
    Returns: list of [total_seq_len] arrays, one per layer.
    """
    per_layer = []
    for attn in attn_tuple:
        a = attn.float().mean(dim=1).squeeze(0).cpu().numpy()  # [q_len, k_len]
        key_probs = a[-1, :total_seq_len]  # last query position
        per_layer.append(key_probs)
    return per_layer


# ── Experiment 3: Single-chunk position shift test ──────────────────────────
def exp3_single_chunk_shift(model, tokenizer, inv_freq, texts, q_short):
    """Single chunk: inject at position 0 (correct) vs 500 (wrong) vs 500+rerot."""
    chunk_kv, chunk_len = compute_chunk_kv(texts[0], model, tokenizer)
    prompt = f"Based on the context, answer: {q_short}"
    total_len = chunk_len  # for baseline at position 0

    results = {}

    # Condition A: inject at pos 0 (correct, no rerot needed)
    cache_0 = build_dynamic_cache([chunk_kv], [chunk_len], model, inv_freq=None)
    attn_a, _ = get_one_token_attn(model, tokenizer, prompt, cache_0)
    results["pos0_correct"] = attn_to_per_token(attn_a, total_len)

    # Condition B: inject at pos 500 (wrong) — pad with zeros
    # We need a bigger cache with 500 dummy tokens. Create dummy KV.
    num_layers, _, num_kv_heads, _, head_dim = chunk_kv.shape
    target_dtype = next(model.parameters()).dtype
    pad_len = 500
    per_layer = []
    for li in range(num_layers):
        chunk_layer = torch.from_numpy(chunk_kv[li].astype(np.float16))
        # Create dummy zeros and insert chunk at offset 500
        k_dummy = torch.zeros(num_kv_heads, pad_len, head_dim, dtype=chunk_layer.dtype)
        v_dummy = torch.zeros(num_kv_heads, pad_len, head_dim, dtype=chunk_layer.dtype)
        k = torch.cat([k_dummy, chunk_layer[0]], dim=1).unsqueeze(0).to(device=model.device, dtype=target_dtype)
        v = torch.cat([v_dummy, chunk_layer[1]], dim=1).unsqueeze(0).to(device=model.device, dtype=target_dtype)
        per_layer.append((k, v))
    from transformers.cache_utils import DynamicCache
    cache_b = DynamicCache(ddp_cache_data=per_layer)
    attn_b, _ = get_one_token_attn(model, tokenizer, prompt, cache_b)
    results["pos500_no_rerot"] = attn_to_per_token(attn_b, pad_len + chunk_len)

    # Condition C: inject at pos 500 WITH re-rotation
    per_layer = []
    for li in range(num_layers):
        k_rerot, v = rerotate_kv(chunk_kv, li, inv_freq, pad_len)  # delta=500
        k_dummy = torch.zeros(num_kv_heads, pad_len, 128, dtype=k_rerot.dtype)
        v_dummy = torch.zeros(num_kv_heads, pad_len, 128, dtype=k_rerot.dtype)
        k = torch.cat([k_dummy, k_rerot], dim=1).unsqueeze(0).to(device=model.device, dtype=target_dtype)
        v = torch.cat([v_dummy, v], dim=1).unsqueeze(0).to(device=model.device, dtype=target_dtype)
        per_layer.append((k, v))
    cache_c = DynamicCache(ddp_cache_data=per_layer)
    attn_c, _ = get_one_token_attn(model, tokenizer, prompt, cache_c)
    results["pos500_rerot"] = attn_to_per_token(attn_c, pad_len + chunk_len)

    return results, chunk_len


# ── Experiment 1: Multi-chunk controlled comparison ──────────────────────────
def exp1_multi_chunk(model, tokenizer, inv_freq, texts, q_short):
    """Multi-chunk: same text, compare true concat vs inject with/without rerot."""
    # Compute KV for each chunk
    all_kvs, chunk_lengths = [], []
    for t in texts:
        kv, cl = compute_chunk_kv(t, model, tokenizer)
        all_kvs.append(kv); chunk_lengths.append(cl)
    total_tokens = sum(chunk_lengths)

    # Condition A: True concatenation (fresh forward pass with concatenated text)
    full_text = "\n\n".join(texts)
    concat_kv, concat_len = compute_chunk_kv(full_text, model, tokenizer)
    prompt = f"Based on the context, answer: {q_short}"
    # For concat, we inject at position 0 (correct since it's a fresh forward pass)
    cache_concat = build_dynamic_cache([concat_kv], [concat_len], model, inv_freq=None)
    attn_a, _ = get_one_token_attn(model, tokenizer, prompt, cache_concat)
    per_token_a = attn_to_per_token(attn_a, concat_len)

    # Condition B: Inject without re-rotation
    cache_b = build_dynamic_cache(all_kvs, chunk_lengths, model, inv_freq=None)
    attn_b, _ = get_one_token_attn(model, tokenizer, prompt, cache_b)
    per_token_b = attn_to_per_token(attn_b, total_tokens)

    # Condition C: Inject WITH re-rotation
    cache_c = build_dynamic_cache(all_kvs, chunk_lengths, model, inv_freq=inv_freq)
    attn_c, _ = get_one_token_attn(model, tokenizer, prompt, cache_c)
    per_token_c = attn_to_per_token(attn_c, total_tokens)

    # Compute per-layer KL and cosine against true concat
    per_layer = []
    num_layers = len(per_token_a)
    for li in range(num_layers):
        pa = per_token_a[li]
        pb = per_token_b[li][:len(pa)]  # Trim to same length
        pc = per_token_c[li][:len(pa)]
        per_layer.append({
            "layer": li,
            "kl_no_rerot": round(_kl(pa, pb), 4),
            "kl_rerot": round(_kl(pa, pc), 4),
            "cos_no_rerot": round(_cosine(pa, pb), 4),
            "cos_rerot": round(_cosine(pa, pc), 4),
        })

    return {
        "n_chunks": len(texts),
        "total_tokens": total_tokens,
        "per_layer": per_layer,
        "mean_kl_no_rerot": round(float(np.mean([l["kl_no_rerot"] for l in per_layer])), 4),
        "mean_kl_rerot": round(float(np.mean([l["kl_rerot"] for l in per_layer])), 4),
        "mean_cos_no_rerot": round(float(np.mean([l["cos_no_rerot"] for l in per_layer])), 4),
        "mean_cos_rerot": round(float(np.mean([l["cos_rerot"] for l in per_layer])), 4),
    }, per_layer


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Controlled RoPE re-rotation experiments")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=30)
    p.add_argument("--short-prompt", default="Based on the context provided, answer: {q}")
    args = p.parse_args()

    raw_cfg = json.loads(Path(args.config).read_text())
    cfg = {**raw_cfg}
    for s in ("indexing", "inference", "training", "background", "sync", "monitoring"):
        cfg.update(raw_cfg.get("addon_config", {}).get(s, {}))

    # Load questions
    base = Path(cfg.get("version_file", "config.json")).parent
    faq_path = base / "faqs.json"
    if faq_path.exists():
        rows = json.loads(faq_path.read_text())
        questions = [{"question": r.get("question",""), "answer": r.get("answer","")} for r in rows]
    else:
        eval_path = base / "eval_heldout_v1.json"
        if eval_path.exists():
            questions = json.loads(eval_path.read_text()).get("items", [])
        else:
            raise ValueError(f"No questions found at {faq_path} or {eval_path}")
    questions = questions[:args.max_samples]
    print(f"Loaded {len(questions)} questions", flush=True)

    # Load model
    model, tokenizer = load_model(cfg)
    inv_freq = get_inv_freq(model)
    print(f"Model loaded. inv_freq: {inv_freq[:5].tolist()}", flush=True)

    exp1_records, exp3_records = [], []
    exp3_total_deltas, exp3_improvements = [], []

    for qi, item in enumerate(questions):
        q = item["question"]
        print(f"\n{'='*60}", flush=True)
        print(f"Q {qi+1}/{len(questions)}: {q[:80]}", flush=True)
        print(f"{'='*60}", flush=True)

        texts = retrieve_chunks(q, cfg)
        if len(texts) < 2:
            print("  Skipping — need 2+ chunks", flush=True)
            continue
        short_prompt = args.short_prompt.format(q=q)

        # Exp 1: Multi-chunk controlled
        print("\n--- Exp 1: Multi-chunk controlled ---", flush=True)
        try:
            exp1_result, per_layer = exp1_multi_chunk(model, tokenizer, inv_freq, texts[:3], q)
            exp1_record = {"question": q, **exp1_result}
            exp1_records.append(exp1_record)
            kl_no = exp1_result["mean_kl_no_rerot"]
            kl_yes = exp1_result["mean_kl_rerot"]
            print(f"  KL no_rerot: {kl_no:.4f}  rerot: {kl_yes:.4f}  improvement: {kl_no - kl_yes:+.4f}", flush=True)
        except Exception as e:
            print(f"  Exp 1 FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

        # Exp 3: Single-chunk position shift
        print("\n--- Exp 3: Single-chunk position shift ---", flush=True)
        try:
            exp3_result, chunk_len = exp3_single_chunk_shift(model, tokenizer, inv_freq, texts, q)
            exp3_records.append({"question": q, "chunk_len": chunk_len, **{
                f"layer_{k}": [float(v) for v in arr]
                for k, arr in exp3_result.items()
            }})
            # Compute KL between conditions
            num_layers = len(exp3_result["pos0_correct"])
            for li in range(num_layers):
                p0 = np.array(exp3_result["pos0_correct"][li])
                p500_no = np.array(exp3_result["pos500_no_rerot"][li])
                p500_yes = np.array(exp3_result["pos500_rerot"][li])
                # Compare relevant portions (the chunk portion, not dummy zeros)
                chunk_start = 500
                chunk_end = 500 + chunk_len
                p_chunk_0 = p0[:chunk_len]  # chunk at pos 0
                p_chunk_500_no = p500_no[chunk_start:chunk_end]  # chunk at pos 500, no rerot
                p_chunk_500_yes = p500_yes[chunk_start:chunk_end]  # chunk at pos 500, rerot
                kl_no = _kl(p_chunk_0, p_chunk_500_no)
                kl_yes = _kl(p_chunk_0, p_chunk_500_yes)
                improvement = kl_no - kl_yes
                if li == 0:
                    print(f"  Layer 0: KL(no_rerot vs correct)={kl_no:.4f}  KL(rerot vs correct)={kl_yes:.4f}  improvement={improvement:+.4f}", flush=True)
                    exp3_total_deltas.append(kl_no)
                    exp3_improvements.append(improvement)
        except Exception as e:
            print(f"  Exp 3 FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    # Aggregation
    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    # Exp 1 summary
    print("\n--- Experiment 1: Multi-chunk controlled ---", flush=True)
    if exp1_records:
        kl_nos = [r["mean_kl_no_rerot"] for r in exp1_records]
        kl_yeses = [r["mean_kl_rerot"] for r in exp1_records]
        improvements = [n - y for n, y in zip(kl_nos, kl_yeses)]
        print(f"  N={len(kl_nos)} questions, {len(exp1_records[0].get('per_layer',[]))} layers", flush=True)
        print(f"  Mean KL no_rerot: {np.mean(kl_nos):.4f}  +/- {np.std(kl_nos, ddof=1)/np.sqrt(len(kl_nos)):.4f}", flush=True)
        print(f"  Mean KL rerot:    {np.mean(kl_yeses):.4f}  +/- {np.std(kl_yeses, ddof=1)/np.sqrt(len(kl_yeses)):.4f}", flush=True)
        print(f"  Mean improvement: {np.mean(improvements):+.4f}  +/- {np.std(improvements, ddof=1)/np.sqrt(len(improvements)):.4f}", flush=True)
        print(f"  Questions where rerot helped: {sum(1 for i in improvements if i > 0)}/{len(improvements)}", flush=True)
        print(f"  Questions where rerot hurt:  {sum(1 for i in improvements if i < 0)}/{len(improvements)}", flush=True)

    # Exp 3 summary
    print("\n--- Experiment 3: Single-chunk position shift ---", flush=True)
    if exp3_total_deltas:
        print(f"  N={len(exp3_total_deltas)} questions", flush=True)
        print(f"  Mean KL(pos500 no_rerot vs pos0): {np.mean(exp3_total_deltas):.4f}", flush=True)
        print(f"  Mean improvement from re-rotation: {np.mean(exp3_improvements):+.4f}", flush=True)
        pct_better = sum(1 for i in exp3_improvements if i > 0) / len(exp3_improvements) * 100
        print(f"  Rerotation helped: {pct_better:.0f}% of cases", flush=True)

    # Save
    output = {
        "config": str(args.config),
        "n_questions": len(questions),
        "experiment1": {
            "per_question": exp1_records if len(exp1_records) < 50 else f"{len(exp1_records)} records (truncated)",
            "aggregate": {
                "mean_kl_no_rerot": round(float(np.mean(kl_nos)), 4) if exp1_records else None,
                "mean_kl_rerot": round(float(np.mean(kl_yeses)), 4) if exp1_records else None,
                "mean_improvement": round(float(np.mean(improvements)), 4) if exp1_records else None,
                "sem_improvement": round(float(np.std(improvements, ddof=1)/np.sqrt(len(improvements))), 4) if exp1_records else None,
                "frac_improved": round(float(sum(1 for i in improvements if i > 0) / len(improvements)), 4) if exp1_records else None,
            } if exp1_records else None,
        },
        "experiment3": {
            "mean_kl_shift_no_rerot": round(float(np.mean(exp3_total_deltas)), 4) if exp3_total_deltas else None,
            "mean_improvement": round(float(np.mean(exp3_improvements)), 4) if exp3_improvements else None,
            "frac_improved": round(sum(1 for i in exp3_improvements if i > 0) / len(exp3_improvements), 4) if exp3_improvements else None,
        } if exp3_total_deltas else None,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, default=str))
    print(f"\n✓ Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
