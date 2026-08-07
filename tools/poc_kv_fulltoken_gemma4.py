"""Proof-of-concept: kv_fulltoken injection on Gemma4.

Run on EC2 with the actual model:
  source ~/kvforge/venv/bin/activate
  CUDA_VISIBLE_DEVICES=<free_gpu> python3 tools/poc_kv_fulltoken_gemma4.py

Tests:
  1. DynamicCache config construction (15 layers, mixed dims)
  2. Single-chunk KV injection with full attention_mask
  3. Multi-chunk KV concatenation injection
  4. Per-layer head_dim verification
"""

import torch
import os

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

from transformers import AutoModelForMultimodalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL_ID = "google/gemma-4-E2B-it"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model_unwrapped():
    """Load Gemma4, delete vision/audio towers, unwrap ClippableLinear."""
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map=DEVICE, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    for name, mod in list(model.model.named_children()):
        if "tower" in name.lower():
            delattr(model.model, name)

    unwrapped = 0
    for name, mod in model.model.language_model.named_modules():
        cls_name = type(mod).__name__
        if "Gemma4ClippableLinear" in cls_name and hasattr(mod, "linear"):
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.model.language_model.named_modules())[parts[0]]
                setattr(parent, parts[1], mod.linear)
                unwrapped += 1
    torch.cuda.empty_cache()
    return model, tokenizer, unwrapped


def build_injected_cache(per_layer_kvs, config):
    """Build DynamicCache from per-layer K/V tuples with correct config."""
    return DynamicCache(config=config, ddp_cache_data=per_layer_kvs)


def generate_with_injected_cache(model, tokenizer, query, cache, cache_seq_len):
    """Generate answer using injected KV cache.

    The critical fix: pass attention_mask covering the FULL sequence
    (cached + new tokens) so _prefill doesn't slice input_ids to empty.
    """
    inputs = tokenizer(query, return_tensors="pt").to(DEVICE)
    query_len = inputs["input_ids"].shape[1]
    full_attn = torch.ones(1, cache_seq_len + query_len, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=full_attn,
            past_key_values=cache,
            max_new_tokens=50,
            do_sample=False,
        )
    answer = tokenizer.decode(out[0][query_len:], skip_special_tokens=True)
    return answer


def generate_text_in_context(model, tokenizer, context, query):
    """Standard text-in-context baseline."""
    messages = [{"role": "user", "content": context + "\n\n" + query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer


# =========================================================================
# Test 1: DynamicCache config construction
# =========================================================================
def test_cache_config_construction():
    print("=" * 60)
    print("Test 1: DynamicCache config construction")
    print("=" * 60)
    model, tokenizer, n_unwrapped = load_model_unwrapped()
    lm = model.model.language_model
    tc = lm.config
    print(f"  Unwrapped {n_unwrapped} ClippableLinear modules")

    cache = DynamicCache(config=tc)
    print(f"  DynamicCache(config=tc) -> {len(cache.layers)} layers")

    n_sliding = sum(1 for l in cache.layers if hasattr(l, "sliding_window"))
    n_full = len(cache.layers) - n_sliding
    print(f"  Sliding: {n_sliding}, Full: {n_full}")
    assert len(cache.layers) == 15, f"Expected 15 layers, got {len(cache.layers)}"
    assert n_sliding == 12, f"Expected 12 sliding, got {n_sliding}"
    assert n_full == 3, f"Expected 3 full, got {n_full}"
    print("  PASSED")
    return model, tokenizer


# =========================================================================
# Test 2: Single-chunk KV injection
# =========================================================================
def test_single_chunk_injection(model, tokenizer):
    print("\n" + "=" * 60)
    print("Test 2: Single-chunk KV injection")
    print("=" * 60)
    lm = model.model.language_model
    tc = lm.config

    context = (
        "The Eiffel Tower is located in Paris, France. "
        "It was built in 1889 by Gustave Eiffel. "
        "The tower is 330 meters tall and is one of the most recognizable "
        "structures in the world."
    )
    query = "How tall is the Eiffel Tower?"

    # Capture KV from context
    ctx_inputs = tokenizer(context, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ctx_out = lm(**ctx_inputs, use_cache=True)
    past_kv = ctx_out.past_key_values
    cache_len = past_kv.layers[0].keys.shape[2]
    print(f"  Context: {ctx_inputs['input_ids'].shape[1]} tokens")
    print(f"  Cache: {len(past_kv.layers)} layers, {cache_len} tokens each")

    # Build injected cache
    cache = DynamicCache(config=tc, ddp_cache_data=[
        (layer.keys, layer.values) for layer in past_kv.layers
    ])

    # Generate with injected KV
    answer_kv = generate_with_injected_cache(
        model, tokenizer, f"Question: {query}\nAnswer:", cache, cache_len
    )
    print(f"  KV Injection: {repr(answer_kv[:200])}")

    # Text-in-context baseline
    answer_text = generate_text_in_context(model, tokenizer, context, f"Question: {query}")
    print(f"  Text Context: {repr(answer_text[:200])}")

    assert len(answer_kv) > 0, "KV injection produced empty answer"
    print("  PASSED")


# =========================================================================
# Test 3: Multi-chunk injection with rerotation
# =========================================================================
def test_multi_chunk_injection(model, tokenizer):
    print("\n" + "=" * 60)
    print("Test 3: Multi-chunk injection with rerotation")
    print("=" * 60)
    lm = model.model.language_model
    tc = lm.config

    chunk1 = (
        "The Eiffel Tower is located in Paris, France. "
        "It was built in 1889 by Gustave Eiffel."
    )
    chunk2 = (
        "The tower is 330 meters tall and is one of the most recognizable "
        "structures in the world. It was the tallest structure in the world until 1930."
    )
    query = "How tall is the Eiffel Tower?"

    def capture_chunk(text):
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = lm(**inputs, use_cache=True)
        return out.past_key_values

    # Capture both chunks and concatenate KV arrays along seq_len
    kv1 = capture_chunk(chunk1)
    kv2 = capture_chunk(chunk2)

    # Concatenate per-layer K/V (both at position 0 — no rerotation)
    concat_kvs = []
    for i in range(15):
        k = torch.cat([kv1.layers[i].keys, kv2.layers[i].keys], dim=2)
        v = torch.cat([kv1.layers[i].values, kv2.layers[i].values], dim=2)
        concat_kvs.append((k, v))

    cache_len = concat_kvs[0][0].shape[2]
    cache = build_injected_cache(concat_kvs, tc)
    print(f"  Combined: {kv1.layers[0].keys.shape[2]} + {kv2.layers[0].keys.shape[2]} = {cache_len} tokens")

    ans = generate_with_injected_cache(
        model, tokenizer, f"Question: {query}\nAnswer:", cache, cache_len
    )
    print(f"  KV (multi-chunk):  {repr(ans[:200])}")

    # Text-in-context baseline
    context = chunk1 + " " + chunk2
    answer_text = generate_text_in_context(model, tokenizer, context, f"Question: {query}")
    print(f"  Text Context:     {repr(answer_text[:200])}")

    assert len(ans) > 0, "Multi-chunk KV injection produced empty answer"
    print("  PASSED")


# =========================================================================
# Test 4: Verify per-layer head_dims in injected cache
# =========================================================================
def test_per_layer_head_dims(model, tokenizer):
    print("\n" + "=" * 60)
    print("Test 4: Verify per-layer head_dims in injected cache")
    print("=" * 60)
    lm = model.model.language_model
    tc = lm.config

    context = "The Eiffel Tower is 330 meters tall."
    ctx_inputs = tokenizer(context, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ctx_out = lm(**ctx_inputs, use_cache=True)

    cache = DynamicCache(config=tc, ddp_cache_data=[
        (layer.keys, layer.values) for layer in ctx_out.past_key_values.layers
    ])

    print(f"  Layer | Type           | K shape                     | hd  | Expected")
    print(f"  ------+----------------+-----------------------------+-----+---------")
    expected = {
        0: 256, 1: 256, 2: 256, 3: 256, 4: 512,
        5: 256, 6: 256, 7: 256, 8: 256, 9: 512,
        10: 256, 11: 256, 12: 256, 13: 256, 14: 512,
    }
    all_ok = True
    for i, layer in enumerate(cache.layers):
        k = layer.keys
        hd = k.shape[-1]
        lt = tc.layer_types[i]
        exp = expected[i]
        ok = "OK" if hd == exp else "MISMATCH"
        print(f"  L{i:2d} | {lt:20s} | {str(list(k.shape)):27s} | {hd:3d} | {exp} {ok}")
        if hd != exp:
            all_ok = False

    assert all_ok, "Per-layer head_dims mismatch"
    print("  PASSED")


def test_integrated_generate_function(model, tokenizer):
    """Test _generate_from_stacked_kv (the patched function) end-to-end."""
    import sys
    _root = os.path.join(os.path.dirname(__file__), "..")
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from pipeline.kv_inference import _generate_from_stacked_kv
    lm = model.model.language_model
    context = ("The Eiffel Tower is located in Paris, France. "
               "It was built in 1889. The tower is 330 meters tall and is "
               "one of the most recognizable structures in the world.")
    ctx_in = tokenizer(context, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ctx_out = lm(**ctx_in, use_cache=True)
    all_kvs = [(layer.keys, layer.values) for layer in ctx_out.past_key_values.layers]
    ans = _generate_from_stacked_kv("How tall is the Eiffel Tower?", all_kvs, model, tokenizer, max_new_tokens=50)
    print(f"  _generate_from_stacked_kv: {repr(ans[:200])}")
    assert "330" in ans or "meters" in ans, f"Should mention height, got: {ans[:100]}"
    print("  PASSED — kv_fulltoken works via integrated function")


if __name__ == "__main__":
    model, tokenizer = test_cache_config_construction()
    test_single_chunk_injection(model, tokenizer)
    test_multi_chunk_injection(model, tokenizer)
    test_per_layer_head_dims(model, tokenizer)
    test_integrated_generate_function(model, tokenizer)
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
