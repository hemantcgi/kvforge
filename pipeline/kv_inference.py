"""
kv_inference.py — KV-injected inference with text-in-context fallback.

Decision logic per query:
  ALL chunks have kv_version == current_lora_version → KV injection (fast)
  ANY chunk stale or null                            → text-in-context fallback
  Either path                                        → enqueue stale chunks for bg heal
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.kv_utils as kv_utils
import pipeline.kv_background as kv_background
import core.model_loader as model_loader
import core.version as ver
from core.kv_utils import deserialize_kv, load_token_kv, load_token_kv_list
from pipeline.bedrock_rag import _run_search, Config
from fastembed import TextEmbedding
from vectorstore.registry import get_store

# Track which query_log_db paths have already been initialised in this process
# so init_db() is not called on every inference request.
_initialized_query_log_dbs: set[str] = set()

# Cache fastembed TextEmbedding instances by model name to avoid reloading the
# ONNX model on every inference call.
_embedder_cache: dict[str, Any] = {}


SYSTEM_PROMPT = (
    "You are a precise assistant. Answer ONLY using the provided context. "
    "Cite sources inline as [page P]. "
    "End with: Confidence: <0-100>%  — <one sentence explanation>"
)


# ── Pure decision functions (testable without GPU) ────────────────────────

def decide_inference_mode(
    chunks: list[dict],
    current_lora_version: int,
    phase: int = 2,
    kds_threshold: float | None = None,
    fkds_threshold: float | None = None,
) -> str:
    """Return 'kv_injection' if all chunks are fresh, cached, and KDS/fKDS-eligible.

    KDS (Knowledge Differentiation Score) and fKDS (factual KDS) eligibility are
    conjunctive: every retrieved chunk must carry the required score at or above
    the configured threshold.  fKDS is preferred when ``fkds_threshold`` is set;
    otherwise the legacy ``kds_threshold`` is used.  A threshold of ``None`` (or
    no calibrated corpus threshold) fails closed to ``text_fallback``.
    """
    if phase < 2:
        return "text_fallback"
    if fkds_threshold is None and kds_threshold is None:
        return "text_fallback"  # no calibrated threshold → fail-closed
    for chunk in chunks:
        v = chunk.get("kv_version")
        if v is None or v < current_lora_version:
            return "text_fallback"
        if chunk.get("kv_cache") is None:
            return "text_fallback"  # kv_cache missing — can't inject
        if fkds_threshold is not None:
            fkds = chunk.get("fkds")
            if fkds is not None and fkds >= fkds_threshold:
                continue
            # fKDS missing or below threshold: fall back to legacy KDS if
            # both thresholds are configured. This keeps older chunks that
            # only have a KDS score eligible when fKDS has not been computed.
            if kds_threshold is not None:
                kds = chunk.get("kds")
                if kds is not None and kds >= kds_threshold:
                    continue
            return "text_fallback"  # chunk lacks fKDS/KDS evidence or is ineligible
        else:
            kds = chunk.get("kds")
            if kds is None or kds < kds_threshold:
                return "text_fallback"  # chunk lacks KDS evidence or is ineligible
    return "kv_injection"


def get_stale_chunk_ids(chunks: list[dict], current_lora_version: int) -> list[int]:
    return [
        c["chunk_id"] for c in chunks
        if c.get("kv_version") is None or c["kv_version"] < current_lora_version
    ]


def route_chunk_injection(chunk: dict, cfg: dict, tq_config=None, vector_store=None) -> dict:
    """Route a single chunk to the correct injection path.

    Returns a dict with keys: path ("enhanced"/"active"/"archive"), kv_arr, text.
    """
    payload = chunk.get("payload", chunk)
    status = payload.get("status", "active")
    kv_token_path = payload.get("kv_token_path")

    if kv_token_path and status != "archived":
        # Check if file uses the variable-head-dim format (Gemma4)
        try:
            _header = np.load(str(kv_token_path), allow_pickle=False)
            is_var_hd = _header.get("_var_hd", np.array(False)).item()
            _header.close()
        except Exception:
            is_var_hd = False
        if is_var_hd:
            arr = load_token_kv_list(kv_token_path)
        else:
            arr = load_token_kv(kv_token_path, tq_config=tq_config)
        return {"path": "enhanced", "kv_arr": arr, "text": None}

    if status == "archived":
        text = _fetch_archive_text(payload.get("archive_path", ""), payload.get("text", ""))
        count = payload.get("archive_retrieval_count", 0) + 1
        if vector_store is not None:
            vector_store.set_payload(
                collection=cfg.get("collection", ""),
                point_id=chunk.get("id"),
                payload={"archive_retrieval_count": count},
            )
        return {"path": "archive", "kv_arr": None, "text": text}

    # Support both flat configs and nested addon_config.indexing.
    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    effective_cfg = {**cfg, **indexing_cfg}
    shape = (effective_cfg["kv_num_layers"], 2, effective_cfg["kv_num_heads"], effective_cfg["kv_head_dim"])
    arr = deserialize_kv(payload["kv_cache"], shape)
    return {"path": "active", "kv_arr": arr, "text": None}


def _fetch_archive_text(archive_path: str, fallback_text: str = "") -> str:
    """Return text from archive_path if it exists, else fallback_text."""
    if not archive_path:
        return fallback_text
    try:
        p = Path(archive_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return fallback_text


# ── Inference paths ───────────────────────────────────────────────────────


def _get_inv_freq(model) -> torch.Tensor | None:
    """Return the model's RoPE inverse-frequency tensor from a text layer.

    Skips vision/audio tower modules (Gemma4's vision tower exposes a
    32-dim inv_freq that mismatches text-layer head_dims).  If no text
    layer has an inv_freq attribute, returns None (rerotation is skipped).
    """
    for name, mod in model.named_modules():
        if any(tag in name for tag in ("vision_tower", "audio_tower", ".vision_", ".audio_")):
            continue
        inv_freq = getattr(mod, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor):
            return inv_freq
    return None


def _rerotate_meanpool_chunks(chunk_kvs: list[np.ndarray], model) -> list[np.ndarray]:
    """Rerotate each mean-pooled chunk's K from position 0 to its global index.

    Mean-pooled chunks are treated as one virtual token per chunk, so chunk
    ``i`` moves from position 0 to position ``i``. This is an approximation
    because the mean-pooled K vector is an average over all in-chunk token
    positions; the alternative (no rerotation) leaves every chunk's K at
    position 0 and produces identical RoPE angles for all chunks.
    """
    inv_freq = _get_inv_freq(model)
    if inv_freq is None:
        return chunk_kvs
    num_layers = chunk_kvs[0].shape[0]
    for i, arr in enumerate(chunk_kvs):
        delta = torch.tensor([float(i)], dtype=torch.float32)
        for layer_idx in range(num_layers):
            k = torch.from_numpy(arr[layer_idx, 0].astype(np.float16))
            k_rot = kv_utils.rerotate_keys(k.unsqueeze(1), inv_freq, delta)
            arr[layer_idx, 0] = k_rot.squeeze(1).numpy().astype(np.float16)
    return chunk_kvs


def _get_rotary_emb_for_model(model):
    """Get the text-model rotary embedding module for Gemma4.

    For models with ``global_head_dim`` (Gemma4), returns the text model's
    ``rotary_emb`` which has per-layer-type ``inv_freq`` buffers.
    Returns None for standard models.
    """
    tc = getattr(model.config, "text_config", model.config)
    if not (hasattr(tc, "global_head_dim") and tc.global_head_dim
            and tc.global_head_dim != getattr(tc, "head_dim", 256)):
        return None
    # Navigate to the text model's rotary embedding
    lm = getattr(getattr(model, "model", None), "language_model", None)
    if lm is None:
        lm = getattr(model, "language_model", None)
    if lm is None:
        lm = getattr(model, "model", None)
    return getattr(lm, "rotary_emb", None)


def _rerotate_fulltoken_chunks(
    chunk_kvs: list[list[np.ndarray]],
    chunk_lengths: list[int],
    model,
) -> list[list[np.ndarray]]:
    """Rerotate each full-token chunk's K to its global position in the cache.

    Each chunk was originally computed at in-chunk positions 0..L-1; when
    concatenated, chunk ``i`` starts at position ``sum(chunk_lengths[:i])``.

    For models with variable head_dim (Gemma4), uses the model's per-layer-type
    rotary_emb to obtain the correct inv_freq (sliding: theta=1e4, full_attention:
    theta=1e6 with partial_rotary_factor=0.25) for each layer.
    """
    rotary_emb = _get_rotary_emb_for_model(model)
    inv_freq = _get_inv_freq(model)
    if inv_freq is None and rotary_emb is None:
        return chunk_kvs

    tc = getattr(model.config, "text_config", model.config)
    layer_types = getattr(tc, "layer_types", None)

    offsets = [0]
    for length in chunk_lengths:
        offsets.append(offsets[-1] + length)
    num_layers = len(chunk_kvs[0])

    for chunk_idx, arr in enumerate(chunk_kvs):
        seq_len = arr[0].shape[1]
        delta = torch.full((seq_len,), float(offsets[chunk_idx]), dtype=torch.float32)

        for layer_idx in range(num_layers):
            k_np = arr[layer_idx][0]  # [num_kv_heads, seq_len, head_dim]
            k = torch.from_numpy(k_np.astype(np.float16))

            if rotary_emb is not None and layer_types is not None:
                lt = layer_types[layer_idx]
                head_dim = k.shape[-1]
                device = k.device
                dummy = torch.zeros(1, seq_len, 1, dtype=k.dtype, device=device)
                cos, sin = rotary_emb(dummy, delta.unsqueeze(0).to(device), lt)
                # cos, sin: [1, seq_len, head_dim]
                cos = cos.squeeze(0)
                sin = sin.squeeze(0)
                half_d = head_dim // 2
                rh = torch.cat([-k[..., half_d:], k[..., :half_d]], dim=-1)
                k_rot = k * cos + rh * sin
            else:
                k_rot = kv_utils.rerotate_keys(k, inv_freq, delta)

            arr[layer_idx][0] = k_rot.numpy().astype(np.float16)
    return chunk_kvs


def generate_with_kv(query: str, chunks: list[dict],
                      model, tokenizer, cfg: dict,
                      extra_context: str = "") -> str:
    """Fast path: inject pre-computed KV tensors as past_key_values."""
    num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)
    kv_shape = (num_layers, 2, num_kv_heads, head_dim)

    chunk_kvs = [
        kv_utils.deserialize_kv(c["kv_cache"], shape=kv_shape)
        for c in chunks
    ]
    # Correct RoPE positions: each mean-pooled chunk was computed at position 0;
    # shift its K to its index in the concatenated virtual sequence.
    chunk_kvs = _rerotate_meanpool_chunks(chunk_kvs, model)
    past_kv = kv_utils.stack_past_key_values(
        chunk_kvs, num_layers=num_layers,
        num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    # Move past_kv tensors to model device.
    # DynamicCache (transformers >= 5.x): move each layer's keys/values in place.
    # Legacy tuple format: rebuild tuple with device-moved tensors.
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            for layer in past_kv.layers:
                layer.keys = layer.keys.to(model.device)
                layer.values = layer.values.to(model.device)
        else:
            past_kv = tuple(
                (k.to(model.device), v.to(model.device)) for k, v in past_kv
            )
    except ImportError:
        past_kv = tuple(
            (k.to(model.device), v.to(model.device)) for k, v in past_kv
        )

    context_prefix = f"Additional context:\n{extra_context}\n\n" if extra_context else ""
    prompt = f"{context_prefix}{SYSTEM_PROMPT}\n\nQuestion: {query}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Explicit position IDs so the query tokens are placed immediately after
    # the injected KV cache instead of starting from position 0.
    prefix_len = past_kv[0][0].shape[2] if isinstance(past_kv, tuple) else past_kv.get_seq_length()
    position_ids = torch.arange(
        prefix_len, prefix_len + inputs["input_ids"].shape[1],
        dtype=torch.long, device=model.device,
    ).unsqueeze(0)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            position_ids=position_ids,
            past_key_values=past_kv,
            max_new_tokens=cfg.get("max_new_tokens", 256),
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)


def generate_text_in_context(query: str, chunks: list[dict],
                               model, tokenizer,
                               max_new_tokens: int = 256,
                               temperature: float = 0.7,
                               top_p: float = 0.9,
                               repetition_penalty: float = 1.2,
                               extra_context: str = "") -> str:
    """Fallback path: include chunk text in prompt."""
    context = "\n\n---\n\n".join(
        f"[page {c['page']}, score {c['score']}]\n{c['text']}"
        for c in chunks
    )
    if extra_context:
        context += f"\n\n---\n\n{extra_context}"
    # Direct instruction prompt — avoids chat-template tokens that confuse the
    # model when used without the exact fine-tuning format.
    prompt = (
        f"Using only the context below, answer the question concisely in 1-2 sentences. "
        f"Cite page numbers.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1600,
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()


def answer_with_mode(
    query: str,
    cfg: dict,
    force_mode: str | None = None,
    use_lora: bool = True,
) -> tuple[str, str]:
    """
    Full inference path with an optional mode override for scientific evaluation.

    Args:
        query: User question.
        cfg: Datasource config dict.
        force_mode: One of ``text_rag``, ``kv_meanpool``, ``kv_fulltoken``,
            ``parametric``, or ``None``.  ``None`` uses the normal dynamic
            decision (fresh KV → KV injection, else text fallback).
        use_lora: If False, load the base model without LoRA.  Use this
            for ``text_rag`` and ``kv_meanpool`` evaluation to measure
            retrieval + reading independently of LoRA training.

    Returns:
        Tuple of (answer_text, mode_used).
    """
    if force_mode is None or force_mode == "parametric":
        # Parametric answers are handled by the caller (prs_evaluator) or by
        # generating directly without retrieved context.  When force_mode is
        # None we fall through to the normal dynamic path.
        pass

    # Support both flat configs and the nested addon_config produced by the
    # top-level config files.
    embed_model = cfg.get("embed_model", cfg.get("addon_config", {}).get("indexing", {}).get("embed_model", "BAAI/bge-small-en-v1.5"))
    if embed_model not in _embedder_cache:
        _embedder_cache[embed_model] = TextEmbedding(model_name=embed_model, show_download_progress=False)
    embedder = _embedder_cache[embed_model]
    store = get_store(cfg)

    # Build Config from cfg dict — use only keys that Config expects.
    # If cfg is nested, prefer the indexing addon values.
    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    flat_cfg = {**cfg, **indexing_cfg, **cfg.get("addon_config", {}).get("inference", {})}
    rag_cfg = Config(**{k: flat_cfg[k] for k in Config.__dataclass_fields__ if k in flat_cfg})

    hits = _run_search(query, embedder, store, rag_cfg)
    if not hits:
        if force_mode == "parametric":
            return "", "parametric"
        return "", "text_rag"

    chunks = [
        {
            "chunk_id": h.id,
            "text": h.payload["text"],
            "page": h.payload.get("page"),
            "score": round(h.score, 4),
            "kv_cache": h.payload.get("kv_cache"),
            "kv_version": h.payload.get("kv_version"),
            "kv_token_path": h.payload.get("kv_token_path"),
            "kds": h.payload.get("kds"),
            "status": h.payload.get("status", "active"),
        }
        for h in hits
    ]

    current_ver = ver.get_lora_version()
    lora_ckpt = cfg.get("checkpoint_path") or ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt if use_lora else None)

    # Record access
    for rank, chunk in enumerate(chunks, start=1):
        kv_background.record_access(chunk["chunk_id"], rank)

    # Enqueue stale for background healing
    stale = get_stale_chunk_ids(chunks, current_ver)
    if stale:
        kv_background.enqueue_kv_recompute(stale)

    # ── Mode decision with override ───────────────────────────────────────────
    kds_threshold = flat_cfg.get("kds_threshold")
    fkds_threshold = flat_cfg.get("fkds_threshold")
    mode = decide_inference_mode(chunks, current_ver, ver.get_phase(), kds_threshold, fkds_threshold)
    # Map legacy dynamic mode strings to the explicit mode names used by the eval.
    mode = {"kv_injection": "kv_meanpool", "text_fallback": "text_rag"}.get(mode, mode)
    if force_mode == "text_rag":
        mode = "text_rag"
    elif force_mode == "kv_meanpool":
        mode = "kv_meanpool"
    elif force_mode == "kv_fulltoken":
        mode = "kv_fulltoken"
    elif force_mode == "parametric":
        return "", "parametric"

    recompute_ratio = float(cfg.get("recompute_ratio", 0.0))

    if mode == "kv_meanpool":
        # Ensure all chunks have a mean-pool KV tensor; otherwise fall back.
        if any(c.get("kv_cache") is None for c in chunks):
            mode = "text_rag"
        elif recompute_ratio > 0.0:
            # Use full-token KV with partial recompute even when the mean-pool
            # KV is present; on-the-fly full-token KV will be computed.
            answer = generate_with_kv_partial_recompute(query, chunks, model, tokenizer, cfg)
        else:
            answer = generate_with_kv(query, chunks, model, tokenizer, cfg)

    if mode == "kv_fulltoken":
        # Prefer enhanced-tier on-disk full-token KV when available; otherwise
        # recompute the full-token KV for the retrieved chunks on the fly.
        # Variable-head-dim models (Gemma4) are handled automatically:
        #   - on-disk: route_chunk_injection detects _var_hd flag and loads
        #     via load_token_kv_list (preserves all 15 layers with mixed dims)
        #   - on-the-fly: compute_per_token_kv_as_list captures all 15 layers
        if any(c.get("kv_token_path") is None for c in chunks):
            routed = []
            for c in chunks:
                inputs = tokenizer(c["text"], return_tensors="pt", truncation=True, max_length=512).to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs, use_cache=True)
                kv_list = kv_utils.compute_per_token_kv_as_list(outputs.past_key_values)
                routed.append({"path": "active", "kv_arr": kv_list, "text": None})
            answer = generate_with_kv_fulltoken(query, routed, model, tokenizer, cfg)
        else:
            # Wrap the enhanced-tier on-disk path via the existing routing helper.
            tq_config = cfg.get("turboquant")
            routed = [route_chunk_injection(c, cfg, tq_config=tq_config, vector_store=store)
                      for c in chunks]
            # If any chunk is archived or missing, we need text fallback.
            if any(r["path"] in ("archive",) or r["kv_arr"] is None for r in routed):
                mode = "text_rag"
            elif recompute_ratio > 0.0:
                answer = generate_with_kv_partial_recompute(query, chunks, model, tokenizer, cfg)
            else:
                answer = generate_with_kv_fulltoken(query, routed, model, tokenizer, cfg)

    if mode == "text_rag":
        answer = generate_text_in_context(query, chunks, model, tokenizer)

    try:
        from pipeline import query_logger as _ql
        _db = cfg.get("query_log_db", "query_log.db")
        if _db not in _initialized_query_log_dbs:
            _ql.init_db(_db)
            _initialized_query_log_dbs.add(_db)
        _ql.log_query(
            db_path=_db,
            query_text=query,
            answer_text=answer,
            routed_to="retrieval" if mode in ("text_rag", "kv_meanpool", "kv_fulltoken") else mode,
            cluster_id=None,
            chunk_id=str(chunks[0]["chunk_id"]) if chunks else None,
        )
    except Exception:
        pass

    return answer, mode


def generate_with_kv_fulltoken(
    query: str,
    routed_chunks: list[dict],
    model,
    tokenizer,
    cfg: dict,
) -> str:
    """Inject full-token KV arrays (Enhanced Tier) as past_key_values.

    Args:
        query: User query string.
        routed_chunks: Output of ``route_chunk_injection`` for each hit.
        model, tokenizer: Loaded model and tokenizer.
        cfg: Datasource config.
    """
    chunk_kvs = [r["kv_arr"] for r in routed_chunks]
    num_layers = len(chunk_kvs[0]) if chunk_kvs else 0
    chunk_lengths = [arr[0].shape[1] for arr in chunk_kvs]
    chunk_kvs = _rerotate_fulltoken_chunks(chunk_kvs, chunk_lengths, model)
    all_kvs = []
    for layer_idx in range(num_layers):
        ks, vs = [], []
        for chunk_kv in chunk_kvs:
            layer = torch.from_numpy(chunk_kv[layer_idx].astype(np.float16))
            ks.append(layer[0])
            vs.append(layer[1])
        k = torch.cat(ks, dim=1).unsqueeze(0)
        v = torch.cat(vs, dim=1).unsqueeze(0)
        all_kvs.append((k, v))

    return _generate_from_stacked_kv(query, all_kvs, model, tokenizer)


def _generate_from_stacked_kv(
    query: str,
    all_kvs: list[tuple[torch.Tensor, torch.Tensor]],
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> str:
    """Build a HuggingFace Cache object from stacked K/V tensors and generate.

    Args:
        query: User query string.
        all_kvs: List of (k, v) tensors, one per layer. Each tensor has shape
            [1, num_kv_heads, total_seq_len, head_dim].
        model, tokenizer: Loaded model and tokenizer.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        Generated answer text.
    """
    from transformers.cache_utils import DynamicCache

    # For models with variable head_dim (Gemma4) / shared KV layers, use
    # config-aware DynamicCache that creates the correct number of cache
    # layers (15 for Gemma4) with proper sliding/full attention types.
    # Padding to num_hidden_layers (35) would inject empty seq_len=0
    # tensors into shared layer slots, causing generate to fail.
    #
    # Pass the top-level model.config (not text_config) so DynamicCache's
    # get_text_config() can unwrap it correctly for multimodal models.
    tc = model.config
    has_shared_layers = (
        hasattr(tc, "text_config")
        and hasattr(tc.text_config, "num_kv_shared_layers")
        and tc.text_config.num_kv_shared_layers > 0
    )
    if has_shared_layers:
        past_kv = DynamicCache(
            config=tc,
            ddp_cache_data=[(k.to(model.device), v.to(model.device)) for k, v in all_kvs],
        )
    else:
        past_kv = DynamicCache(ddp_cache_data=[
            (k.to(model.device), v.to(model.device)) for k, v in all_kvs
        ])

    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {query}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    cache_len = all_kvs[0][0].shape[2] if all_kvs else 0
    query_len = inputs["input_ids"].shape[1]

    # CRITICAL: Pass attention_mask covering the FULL sequence (cache + query).
    # Without this, HuggingFace's _prefill sees past_length > input_length and
    # slices input_ids to empty (next_sequence_length = query_len - cache_len
    # when input_ids.shape[1] == attention_mask.shape[1]).
    full_attn = torch.ones(1, cache_len + query_len, device=model.device, dtype=torch.long)

    # Explicit position_ids so query tokens are placed immediately after the
    # injected KV cache. The meanpool path (generate_with_kv) also does this.
    position_ids = torch.arange(
        cache_len, cache_len + query_len,
        dtype=torch.long, device=model.device,
    ).unsqueeze(0)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=full_attn,
            position_ids=position_ids,
            past_key_values=past_kv,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)


def _select_recompute_tokens(deviation: np.ndarray, ratio: float) -> np.ndarray:
    """Return a boolean mask of tokens to recompute.

    Selects the top ``ratio`` fraction of tokens by deviation. Always selects
    at least the first token (attention sink) and one token when ratio > 0.

    Args:
        deviation: Per-token deviation array [seq_len].
        ratio: Fraction of tokens to recompute in [0, 1].

    Returns:
        Boolean mask of length ``seq_len``.
    """
    seq_len = deviation.shape[0]
    if ratio <= 0.0:
        return np.zeros(seq_len, dtype=bool)
    if ratio >= 1.0:
        return np.ones(seq_len, dtype=bool)

    n = max(1, int(seq_len * ratio))
    # Always include the attention sink (first token) in the recompute set.
    mask = np.zeros(seq_len, dtype=bool)
    mask[0] = True
    n = max(0, n - 1)
    if n > 0:
        top_k = np.argpartition(deviation, -n)[-n:]
        mask[top_k] = True
    return mask


def generate_with_kv_partial_recompute(
    query: str,
    chunks: list[dict],
    model,
    tokenizer,
    cfg: dict,
) -> str:
    """CacheBlend-style partial KV recompute for KV injection.

    Loads (or computes on-the-fly) full-token KV arrays for each retrieved chunk,
    concatenates them, computes a fresh KV pass over the assembled context, and
    blends the highest-deviation tokens back into the cached KV before injection.

    * ``recompute_ratio=0.0`` is equivalent to full-token KV injection.
    * ``recompute_ratio=1.0`` is equivalent to text-in-context (recompute everything).
    * Intermediate values trade latency for quality.

    Args:
        query: User query string.
        chunks: Retrieved chunk dicts with ``text`` and optionally
            ``kv_token_path`` or ``kv_cache``.
        model, tokenizer: Loaded model and tokenizer.
        cfg: Datasource config containing ``recompute_ratio``.
    """
    ratio = float(cfg.get("recompute_ratio", 0.0))
    if ratio >= 1.0:
        return generate_text_in_context(query, chunks, model, tokenizer)

    # Collect full-token KV arrays for each chunk.
    chunk_kvs: list[np.ndarray] = []
    chunk_texts: list[str] = []
    for c in chunks:
        text = c.get("text", "")
        chunk_texts.append(text)
        if c.get("kv_token_path"):
            arr = load_token_kv(c["kv_token_path"], tq_config=cfg.get("turboquant"))
        else:
            # Compute on-the-fly; this costs one forward pass per chunk but lets
            # partial recompute work on collections that only have mean-pool KV.
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, use_cache=True)
            arr = kv_utils.compute_per_token_kv(outputs.past_key_values)
        chunk_kvs.append(arr)

    # Align cached KV positions with the assembled fresh pass. Each chunk's
    # cached KV was computed at in-chunk positions 0..L-1; when concatenated
    # they must be rerotated to global positions to avoid spurious RoPE
    # deviation during blending.
    chunk_lengths = [arr.shape[3] for arr in chunk_kvs]
    chunk_kvs = _rerotate_fulltoken_chunks(chunk_kvs, chunk_lengths, model)

    # Concatenate cached KVs along the sequence dimension.
    cached_kv = np.concatenate(chunk_kvs, axis=-2)  # [L, 2, H, total_seq_len, d]

    # Build a fresh KV pass over the assembled context using the exact same
    # per-chunk token sequence as the cached KVs. Concatenating the individual
    # tokenizations avoids the token mismatches (extra separators, duplicate
    # BOS tokens) caused by tokenizing the joined text as a single sequence.
    per_chunk_inputs = [
        tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        for text in chunk_texts
    ]
    fresh_input_ids = torch.cat([inp["input_ids"] for inp in per_chunk_inputs], dim=1)
    fresh_attention_mask = torch.cat([inp["attention_mask"] for inp in per_chunk_inputs], dim=1)
    max_length = 2048
    if fresh_input_ids.shape[1] > max_length:
        fresh_input_ids = fresh_input_ids[:, :max_length]
        fresh_attention_mask = fresh_attention_mask[:, :max_length]
    fresh_inputs = {
        "input_ids": fresh_input_ids.to(model.device),
        "attention_mask": fresh_attention_mask.to(model.device),
    }
    with torch.no_grad():
        fresh_outputs = model(**fresh_inputs, use_cache=True)
    fresh_kv = kv_utils.compute_per_token_kv(fresh_outputs.past_key_values)

    # If shapes differ due to truncation, fall back to the fresh KV entirely.
    if fresh_kv.shape[-2] != cached_kv.shape[-2]:
        return _generate_from_full_token_kv(query, fresh_kv, model, tokenizer)

    # Compute per-token deviation at the middle layer.
    mid_layer = cached_kv.shape[0] // 2
    # Shape at mid_layer: [2, H, seq_len, d]
    diff = cached_kv[mid_layer].astype(np.float32) - fresh_kv[mid_layer].astype(np.float32)
    # Per-token L2 norm over (kv_slot, head, head_dim), leaving shape [seq_len].
    deviation = np.sqrt(np.sum(diff ** 2, axis=(0, 1, 3)))

    mask = _select_recompute_tokens(deviation, ratio)
    # Blend: keep cached KV for unselected tokens; use fresh KV for selected tokens.
    blended_kv = cached_kv.copy()
    blended_kv[:, :, :, mask, :] = fresh_kv[:, :, :, mask, :]

    # Build per-layer stacked tensors for _generate_from_stacked_kv.
    num_layers = blended_kv.shape[0]
    all_kvs = []
    for layer_idx in range(num_layers):
        layer = torch.from_numpy(blended_kv[layer_idx].astype(np.float16))
        k = layer[0].unsqueeze(0)  # [1, H, seq_len, d]
        v = layer[1].unsqueeze(0)
        all_kvs.append((k, v))

    return _generate_from_stacked_kv(query, all_kvs, model, tokenizer)


def _generate_from_full_token_kv(
    query: str,
    kv: np.ndarray,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> str:
    """Generate from a single full-token KV array [L, 2, H, seq_len, d]."""
    all_kvs = []
    for layer_idx in range(kv.shape[0]):
        layer = torch.from_numpy(kv[layer_idx].astype(np.float16))
        k = layer[0].unsqueeze(0)
        v = layer[1].unsqueeze(0)
        all_kvs.append((k, v))
    return _generate_from_stacked_kv(query, all_kvs, model, tokenizer, max_new_tokens=max_new_tokens)


def answer_with_retrieval(query: str, cfg: dict) -> str:
    """
    Full SP3 pipeline: search → version check → KV inject or text fallback.
    Called by prs_evaluator.py for RAG-mode answers.
    """
    answer, _ = answer_with_mode(query, cfg, force_mode=None)
    return answer


def route_query(query: str, cfg: dict) -> list[dict]:
    """Dynamic PRS cluster-aware retrieval.

    Embeds *query*, finds its nearest cluster, then calls
    ``answer_with_retrieval`` restricted to chunks from that cluster.
    Falls back to full-collection retrieval when no cluster data exists.

    Args:
        query: User query string.
        cfg: Datasource config dict.

    Returns:
        List of chunk dicts from the nearest cluster (or full collection).
    """
    from pathlib import Path as _Path
    cluster_file = _Path(cfg.get("checkpoint_dir", ".")) / "clusters.json"
    if not cluster_file.exists():
        return []

    try:
        from core.cluster_manager import load_clusters, nearest_cluster
        cluster_data = load_clusters(str(cluster_file))
        embedder = TextEmbedding(model_name=cfg["embed_model"],
                                  show_download_progress=False)
        q_vec = list(embedder.embed([query]))[0]
        import numpy as np
        cluster_id = nearest_cluster(
            np.array(q_vec), np.array(cluster_data["centroids"])
        )
        store = get_store(cfg)
        from pipeline.bedrock_rag import Config
        rag_cfg = Config(**{k: cfg[k] for k in Config.__dataclass_fields__ if k in cfg})
        hits = store.query(
            cfg["collection"], q_vec.tolist(), top_k=cfg.get("top_k", 5),
            scroll_filter={"cluster_id": str(cluster_id)},
        )
        return [
            {
                "chunk_id": h.id,
                "text": h.payload.get("text", ""),
                "page": h.payload.get("page"),
                "score": round(h.score, 4),
                "kv_cache": h.payload.get("kv_cache"),
                "kv_version": h.payload.get("kv_version"),
                "kds": h.payload.get("kds"),
                "cluster_id": str(cluster_id),
            }
            for h in hits
        ]
    except Exception:
        return []


def main() -> None:
    """Pipe-compatible: read JSON from stdin (from bedrock_rag.py search)."""
    if sys.stdin.isatty():
        print('Usage: python3 bedrock_rag.py search "query" | python3 kv_inference.py')
        sys.exit(1)

    data = json.load(sys.stdin)
    query = data["query"]
    chunks = data["chunks"]

    with open("my_config.json") as f:
        cfg = json.load(f)

    kv_background.start(cfg)

    current_ver = ver.get_lora_version()
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)

    for rank, chunk in enumerate(chunks, start=1):
        kv_background.record_access(chunk["chunk_id"], rank)

    stale = get_stale_chunk_ids(chunks, current_ver)
    if stale:
        kv_background.enqueue_kv_recompute(stale)

    mode = decide_inference_mode(
        chunks, current_ver,
        kds_threshold=cfg.get("kds_threshold"),
        fkds_threshold=cfg.get("fkds_threshold"),
    )
    print(f"Mode: {mode}  |  lora_version={current_ver}  |  "
          f"stale_chunks={len(stale)}/{len(chunks)}")
    print("-" * 62)

    if mode == "kv_injection":
        answer = generate_with_kv(query, chunks, model, tokenizer, cfg)
    else:
        answer = generate_text_in_context(query, chunks, model, tokenizer)

    print(answer)


if __name__ == "__main__":
    main()
