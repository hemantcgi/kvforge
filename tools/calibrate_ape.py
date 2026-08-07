"""APE calibration v2: simpler — uses answer_with_mode pipeline for retrieval,
hooks into model for attention capture. No duplicated retrieval logic.

Usage: CUDA_VISIBLE_DEVICES=0 python3 tools/calibrate_ape.py \
         --config examples/usecase4_bedrock_userguide/config.json \
         --num-queries 20 --output ape_calibration.json
"""

import torch, os, sys, json, argparse, time, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--num-queries", type=int, default=20)
    p.add_argument("--output", default="ape_calibration.json")
    p.add_argument("--min-temp", type=float, default=0.3)
    p.add_argument("--max-temp", type=float, default=3.0)
    p.add_argument("--num-temps", type=int, default=20)
    return p.parse_args()


def load_calibration_queries(cfg, num_queries):
    """Get queries from the eval set, with retrieved chunks."""
    with open(cfg["config"]) as f:
        cfg_data = json.load(f)

    eval_path = cfg_data.get("eval_set", "examples/usecase4_bedrock_userguide/eval_tier_500.json")
    if not os.path.isabs(eval_path):
        # Join with the repo root dir (where the config file lives relative to)
        repo_root = os.path.join(os.path.dirname(os.path.abspath(cfg["config"])), "..", "..")
        eval_path = os.path.join(repo_root, eval_path)
    with open(eval_path) as f:
        eval_data = json.load(f)

    items = eval_data.get("items", eval_data)
    return [item["question"] for item in items[:num_queries]]


def get_chunks_for_query(query, cfg_data):
    """Retrieve top-3 chunks using the same pipeline as answer_with_mode."""
    from fastembed import TextEmbedding
    from vectorstore.registry import get_store
    from pipeline.bedrock_rag import _run_search, Config

    embed_model = cfg_data.get("embed_model", cfg_data.get("addon_config", {}).get("indexing", {}).get("embed_model", "BAAI/bge-small-en-v1.5"))
    embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)
    store = get_store(cfg_data)

    indexing_cfg = cfg_data.get("addon_config", {}).get("indexing", {})
    inference_cfg = cfg_data.get("addon_config", {}).get("inference", {})
    flat_cfg = {**cfg_data, **indexing_cfg, **inference_cfg}
    valid_keys = Config.__dataclass_fields__.keys() if hasattr(Config, "__dataclass_fields__") else []
    rag_cfg = Config(**{k: flat_cfg[k] for k in valid_keys if k in flat_cfg})

    hits = _run_search(query, embedder, store, rag_cfg)
    return [{"text": h.payload["text"]} for h in hits[:3]] if hits else [
        {"text": "This is a test chunk. The Eiffel Tower is 330 meters tall."}
    ]


def capture_attn_text_rag(model, tokenizer, query, chunks, cache_len):
    """Run text_rag with concatenated chunk text and capture attention."""
    lm = model.model.language_model
    context = " ".join(c["text"] for c in chunks)
    prompt = f"Question: {query}\nAnswer:"
    # Tokenize context separately for ctx_len
    ctx_ids = tokenizer(context, return_tensors="pt")["input_ids"]
    ctx_len = ctx_ids.shape[1]
    # Full input: context + query_prompt
    full_text = context + " " + prompt
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1600).to(model.device)

    with torch.no_grad():
        outputs = lm(**inputs, use_cache=True, output_attentions=True)

    total_len = inputs["input_ids"].shape[1]
    result = []
    for layer_attn in outputs.attentions:
        if layer_attn is None: continue
        # Attention of query tokens (after context) over context tokens
        # Pad to cache_len if needed (tokenization differences add 1-2 tokens)
        a = layer_attn[0, :, ctx_len:, :ctx_len].cpu()
        if a.shape[-1] < cache_len:
            pad = torch.zeros(a.shape[0], a.shape[1], cache_len - a.shape[-1])
            a = torch.cat([a, pad], dim=-1)
        elif a.shape[-1] > cache_len:
            a = a[:, :, :cache_len]
        result.append(a)
    return result


def capture_attn_kv_fulltoken(model, tokenizer, query, chunks):
    """Run kv_fulltoken with same chunks, capture attention."""
    import core.kv_utils as kv_utils
    from pipeline.kv_inference import _rerotate_fulltoken_chunks
    from transformers.cache_utils import DynamicCache

    lm = model.model.language_model
    tc = model.config

    chunk_kvs_list, chunk_lengths = [], []
    for c in chunks:
        inp = tokenizer(c["text"], return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = lm(**inp, use_cache=True)
        chunk_kvs_list.append(kv_utils.compute_per_token_kv_as_list(out.past_key_values))
        chunk_lengths.append(inp["input_ids"].shape[1])

    chunk_kvs_list = _rerotate_fulltoken_chunks(chunk_kvs_list, chunk_lengths, model)

    num_layers = len(chunk_kvs_list[0])
    all_kvs = []
    for li in range(num_layers):
        ks = [torch.from_numpy(c[li][0].astype(np.float16)) for c in chunk_kvs_list]
        vs = [torch.from_numpy(c[li][1].astype(np.float16)) for c in chunk_kvs_list]
        all_kvs.append((torch.cat(ks, dim=1).unsqueeze(0), torch.cat(vs, dim=1).unsqueeze(0)))

    cache_len = all_kvs[0][0].shape[2]
    q_text = f"Question: {query}\nAnswer:"
    q_inp = tokenizer(q_text, return_tensors="pt").to(model.device)
    q_len = q_inp["input_ids"].shape[1]

    full_attn = torch.ones(1, cache_len + q_len, device=model.device, dtype=torch.long)
    pos_ids = torch.arange(cache_len, cache_len + q_len, dtype=torch.long, device=model.device).unsqueeze(0)

    has_shared = hasattr(tc, "text_config") and hasattr(tc.text_config, "num_kv_shared_layers") and tc.text_config.num_kv_shared_layers > 0
    dk = {"ddp_cache_data": [(k.to(model.device), v.to(model.device)) for k, v in all_kvs]}
    if has_shared: dk["config"] = model.config
    past_kv = DynamicCache(**dk)

    with torch.no_grad():
        outputs = lm(input_ids=q_inp["input_ids"], attention_mask=full_attn, position_ids=pos_ids, past_key_values=past_kv, use_cache=True, output_attentions=True)

    result = []
    for layer_attn in outputs.attentions:
        if layer_attn is None: continue
        a = layer_attn[0, :, :, :cache_len].cpu()
        result.append(a)
    return result, cache_len


def fit_temperature(fresh_list, injected_list, min_temp, max_temp, num_temps):
    """Fit per-layer temperature by averaging KL across queries.

    Each query has different context lengths, so we compute per-query KL
    and average the temperature that minimizes the mean KL across queries.
    """
    temps = np.linspace(min_temp, max_temp, num_temps)
    num_layers = min(len(fresh_list[0]), 15)
    params = {}

    for li in range(num_layers):
        # Collect per-query attention pairs for this layer
        pairs = [
            (q[li], injected_list[qi][li])
            for qi, q in enumerate(fresh_list)
            if li < len(q) and q[li] is not None
            and li < len(injected_list[qi]) and injected_list[qi][li] is not None
        ]
        if not pairs:
            params[f"layer_{li}"] = {"temperature": 1.0}
            continue

        # Find best temperature that minimizes mean KL across all queries
        best_kl, best_t = float("inf"), 1.0
        for temp in temps:
            kls = []
            for f_attn, i_attn in pairs:
                # Flatten each query's attention
                f_prob = torch.softmax(f_attn.float().reshape(-1, f_attn.shape[-1]), dim=-1)
                i_prob = torch.softmax(i_attn.float().reshape(-1, i_attn.shape[-1]) * temp, dim=-1)
                kl = (f_prob * (torch.log(f_prob + 1e-10) - torch.log(i_prob + 1e-10))).sum(dim=-1).mean().item()
                kls.append(kl)
            mean_kl = sum(kls) / len(kls)
            if mean_kl < best_kl:
                best_kl, best_t = mean_kl, float(temp)

        params[f"layer_{li}"] = {"temperature": round(best_t, 4), "kl_divergence": round(best_kl, 6)}
        print(f"  layer_{li}: temp={best_t:.3f} KL={best_kl:.4f} ({len(pairs)} queries)")

    return params


def main():
    args = parse_args()
    cfg_path = os.path.abspath(args.config)

    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["attn_implementation"] = "eager"
    cfg["config"] = cfg_path

    import core.model_loader as model_loader
    model_loader.init(cfg)
    model, tokenizer = model_loader.load(cfg.get("checkpoint_path"))

    print("Model loaded with eager attention")

    queries = load_calibration_queries(cfg, args.num_queries)
    print(f"Calibrating on {len(queries)} queries\n")

    all_fresh, all_injected = [], []
    for qi, query in enumerate(queries):
        print(f"[{qi + 1}/{len(queries)}] {query[:80]}...")
        try:
            chunks = get_chunks_for_query(query, cfg)
            if not chunks:
                print("  No chunks, skipping")
                continue

            t0 = time.time()
            ia, cache_len = capture_attn_kv_fulltoken(model, tokenizer, query, chunks)
            print(f"  kv_fulltoken: {len(ia)} layers, cache={cache_len}, {time.time() - t0:.1f}s")

            t0 = time.time()
            fa = capture_attn_text_rag(model, tokenizer, query, chunks, cache_len)
            # Pad/truncate both to match exactly
            min_len = min(cache_len, fa[0].shape[-1]) if fa and fa[0] is not None else cache_len
            fa = [f[:, :, :min_len] for f in fa]
            ia = [i[:, :, :min_len] for i in ia]
            print(f"  text_rag: {len(fa)} layers, aligned={min_len}, {time.time() - t0:.1f}s")

            all_fresh.append(fa)
            all_injected.append(ia)
        except Exception as e:
            import traceback
            traceback.print_exc()

    if not all_fresh:
        print("No successful calibrations. Exiting.")
        return

    print(f"\nFitting temperature across {len(all_fresh)} queries...\n")
    params = fit_temperature(all_fresh, all_injected, args.min_temp, args.max_temp, args.num_temps)

    with open(args.output, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
