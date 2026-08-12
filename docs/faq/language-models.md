# Language Models

← [Back to FAQ index](../../FAQ.md)

---

### How do I use my own LLM for KV computation?

KVForge uses HuggingFace `AutoModelForCausalLM` for KV computation and LoRA training. Any decoder-only transformer hosted on HuggingFace Hub (or locally) works.

#### Step 1 — Set `llm_model` in your config

```json
{
  "llm_model": "mistralai/Mistral-7B-Instruct-v0.3"
}
```

Other tested models:

```json
{ "llm_model": "google/gemma-4-E2B-it" }
{ "llm_model": "google/gemma-2-2b-it" }
{ "llm_model": "Qwen/Qwen2.5-3B-Instruct" }
{ "llm_model": "microsoft/phi-3-mini-4k-instruct" }
{ "llm_model": "mistralai/Mistral-7B-Instruct-v0.3" }
{ "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0" }
```

`google/gemma-4-E2B-it` is the reference model for KVForge's crossover evaluation — a 2B-parameter model that, when LoRA-fine-tuned on cloud-generated QA pairs, matches or exceeds text-in-context RAG on three of four enterprise corpora.

#### Step 2 — Verify KV shape auto-discovery

KVForge reads `num_hidden_layers`, `num_key_value_heads`, and `hidden_size` / `head_dim` from the HuggingFace model config automatically. Verify before running the full pipeline:

```python
import model_loader

cfg = {"llm_model": "mistralai/Mistral-7B-Instruct-v0.3"}
model_loader.init(cfg)
num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)
print(f"KV shape: layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}")
# Expected for Mistral-7B: layers=32, kv_heads=8, head_dim=128
```

#### Step 3 — Confirm LoRA target modules

KVForge auto-detects which attention projection module names exist in your model and warns if none of the configured names are found:

```python
import model_loader, warnings

cfg = {"llm_model": "mistralai/Mistral-7B-Instruct-v0.3",
       "lora_target_modules": ["q_proj", "k_proj", "v_proj"]}
model_loader.init(cfg)
model, _ = model_loader.load(None)
matched = model_loader.detect_lora_targets(model, cfg["lora_target_modules"])
print("LoRA targets found:", matched)
```

Common LoRA target patterns by model family:

| Model family | Typical `lora_target_modules` |
|-------------|-------------------------------|
| Llama / Mistral / Phi | `["q_proj", "k_proj", "v_proj"]` |
| Falcon | `["query_key_value"]` |
| GPT-2 / GPT-J | `["c_attn"]` |
| BLOOM | `["query_key_value", "dense"]` |
| Gemma | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| Qwen2 | `["q_proj", "k_proj", "v_proj", "o_proj"]` |

For an unknown architecture, list all module names and pick attention projections:

```python
for name, module in model.named_modules():
    if "proj" in name or "attn" in name:
        print(name)
```

#### Step 4 — Use a locally saved model

If your model is saved to disk rather than hosted on HuggingFace:

```json
{
  "llm_model": "/home/ubuntu/models/my-fine-tuned-llama"
}
```

`model_loader.py` passes `llm_model` directly to `AutoModelForCausalLM.from_pretrained()`, which accepts both Hub IDs and local paths.

#### Step 5 — Run the pipeline

```bash
python index_and_train.py my_document.pdf \
  --config datasource_my-corpus.json \
  --faqs my_faqs.json
```

---

### How do I use a gated model like Llama 3 that requires a HuggingFace token?

Models like `meta-llama/Llama-3.2-3B-Instruct` require you to:
1. Create a HuggingFace account at [huggingface.co](https://huggingface.co)
2. Visit the model card and click **Agree and access repository**
3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **Read** permission

#### Option A — Config file (recommended for servers)

```json
{
  "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
  "hf_token":  "hf_xxxxxxxxxxxxxxxxxxxx"
}
```

KVForge sets `HF_TOKEN` in the environment before calling `from_pretrained()`.

> **Security note:** Do not commit `datasource_*.json` files containing tokens to version control. Add `datasource_*.json` to `.gitignore` or use the environment variable approach instead.

#### Option B — Environment variable (recommended for CI/CD)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
python index_and_train.py document.pdf --config my_config.json
```

Or in a shell script:

```bash
#!/bin/bash
export HF_TOKEN=$(cat ~/.hf_token)   # read from a file not in version control
python index_and_train.py "$@"
```

#### Option C — HuggingFace CLI login (interactive)

```bash
pip install huggingface_hub
huggingface-cli login
# Enter your token when prompted — it is saved to ~/.cache/huggingface/token
```

After login, no token is needed in the config or environment.

#### Verifying access

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct",
                                     token="hf_xxxx")
print(config.model_type)   # should print: llama
```

If you see `401 Client Error: Unauthorized`, your token is invalid or you have not accepted the license agreement.

---

### Can I use an API-hosted LLM (OpenAI, Anthropic, Gemini)?

#### What requires a local model

KV cache computation and LoRA fine-tuning both require direct access to the model's internal tensors. Specifically:

- **KV computation** needs `outputs.past_key_values` — the raw attention tensors produced during a forward pass. No API exposes this.
- **LoRA training** needs gradient flow through the model's weight matrices. No API exposes this.

These operations cannot be performed against API-hosted models. A local HuggingFace model is required for Phases 2 and 3.

#### What can use an API LLM

Phase 1 text-in-context fallback is just prompt engineering — retrieved chunks are placed into a prompt and the model generates an answer. You can replace the local generation in `kv_inference.generate_text_in_context()` with an API call:

```python
# kv_inference.py — replace generate_text_in_context() for API usage
import openai

def generate_text_in_context_openai(query: str, chunks: list[dict],
                                     api_key: str,
                                     model: str = "gpt-4o-mini") -> str:
    context = "\n\n---\n\n".join(
        f"[page {c['page']}, score {c['score']}]\n{c['text']}"
        for c in chunks
    )
    prompt = (
        f"Using only the context below, answer the question in 2-4 sentences. "
        f"Cite page numbers.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
```

This gives you a fully functional **Phase 1 RAG system** with no GPU required. KV injection (Phase 2) and parametric answering (Phase 3) remain unavailable with API models.

#### Practical hybrid architecture

If you want the best of both worlds:

```
Retrieval:   OpenAI text-embedding-3-small  (high-quality embeddings, no GPU)
Answer:      OpenAI gpt-4o-mini             (Phase 1, no GPU)
KV compute:  local Gemma 4 2B               (Phase 2+, GPU required)
LoRA train:  local Gemma 4 2B               (Phase 2+, GPU required)
```

Use OpenAI embeddings for the highest retrieval quality, and the local small model only for the KV/LoRA work that requires tensor access.

---

### Can I run this without a GPU?

The table below shows which operations are CPU-compatible:

| Operation | CPU | GPU | Notes |
|-----------|:---:|:---:|-------|
| `kvforge.py init` | ✅ | ✅ | Config scaffolding only |
| `kvforge.py index` | ✅ | ✅ | Embedding runs on CPU with FastEmbed |
| `kvforge.py search` | ✅ | ✅ | Embedding + vector search |
| `python -m pytest tests/` | ✅ | ✅ | All 76 tests mock GPU modules |
| `monitoring_dashboard.py` | ✅ | ✅ | FastAPI dashboard |
| `prs_evaluator.py` (evaluation only) | ✅ | ✅ | If using API LLM for generation |
| KV tensor computation | ❌ | ✅ | LLM forward pass required |
| LoRA training | ❌ | ✅ | Gradient computation required |
| KV injection at query time | ❌ | ✅ | Tensor operations on model device |

#### Running the test suite on CPU

```bash
# All 76 tests pass on CPU — GPU modules are mocked
python -m pytest tests/ -v --override-ini="addopts="
```

Expected time: ~15–30 seconds on a modern laptop.

#### Local development workflow without a GPU

1. Use `kvforge.py init / index / search` for all ingestion and retrieval work
2. Use the dashboard to verify indexing output
3. When ready to train, push the config and data to a GPU server:

```bash
rsync -avz --exclude='venv/' --exclude='__pycache__/' \
  -e "ssh -i your-key.pem" \
  ./ ubuntu@<gpu-server>:~/kvforge/

ssh -i your-key.pem ubuntu@<gpu-server>
cd ~/kvforge
python index_and_train.py document.pdf --config datasource_my-corpus.json --faqs faqs.json
```

---

← [Back to FAQ index](../../FAQ.md)
