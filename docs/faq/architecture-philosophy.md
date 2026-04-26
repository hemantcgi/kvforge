# Architecture Philosophy & Battlecards

← [Back to FAQ index](../../FAQ.md)

This section addresses the "why" questions — architectural trade-offs, design decisions, and the reasoning behind KVForge's approach. These are the questions that come up when evaluating KVForge against simpler alternatives. Use these as a reference when justifying the approach to technical audiences.

---

### Why fine-tune at all? Can't we just store KV tensors for every chunk and use the vector database as a live KV cache at query time?

**The short answer:** Phase 2 already does exactly this — and it is a genuine, production-worthy approach that gives ~4–5× speedup over standard RAG. Phase 3 fine-tuning solves a different and deeper set of problems that KV injection cannot address.

#### What Phase 2 already does

KVForge Phase 2 IS the approach described in this question. At index time, every chunk runs a forward pass through the model; the resulting KV tensors are stored in the vector database. At query time, the top-K matching chunks are retrieved, their KV tensors are loaded, and injected directly into the model's attention layers via `past_key_values`. The model never re-encodes the chunk text — it attends over the pre-computed KV state instead.

This gives roughly 4–5× latency reduction over Phase 1 for the same retrieval quality. For many use cases, Phase 2 is the right stopping point and Phase 3 is optional.

#### The three problems KV injection cannot solve

**1. Retrieval dependence**

KV injection makes reading the context faster — it does not make finding the right context more reliable. The model still depends on vector similarity search to retrieve the right chunks. If the query is a paraphrase, a multi-hop question, or uses vocabulary that does not align with the embedding space, the wrong chunks are retrieved and injected. The model can only answer what is in front of it.

Fine-tuning moves knowledge from "retrieved context" into model weights. After fine-tuning, the model can answer queries about the domain without retrieving anything — the answer is in the weights regardless of how the question is phrased.

**2. Memory: you cannot fit all chunk KV tensors into GPU VRAM simultaneously**

This is a hard physics constraint. KVForge stores KV tensors at shape `[num_layers, 2, num_kv_heads, head_dim]` — already compressed by mean-pooling the sequence dimension. For Llama 3.2 3B, each chunk's KV tensor is ~115 KB. With 2,520 chunks, that is ~290 MB of payload in Qdrant — manageable for storage.

But GPU VRAM is a different story. You cannot load all 2,520 chunk KV tensors into attention memory at once. At query time you load only the top-K retrieved chunks. This means the model sees a small window of the corpus per query and cannot reason holistically across all chunks simultaneously.

Fine-tuning trains across all chunks together during the training pass. The resulting LoRA weights encode relationships between concepts spread across different chunks — multi-hop reasoning that no retrieval window can replicate.

**3. KV tensors are tied to the exact model weights that computed them**

Every time the base model changes — LoRA checkpoint update, quantization change, adapter swap — every stored KV tensor in the vector database becomes stale and must be recomputed. KVForge tracks this via `kv_version` and runs a background healing daemon, but at large scale (hundreds of thousands of chunks) the recomputation load becomes a perpetual background job.

Fine-tuning bakes domain knowledge into the LoRA adapter once. The adapter is stable until explicitly retrained. No background recomputation loop is needed for routine queries.

#### The fundamental difference between Phase 2 and Phase 3

| | Phase 2 — KV Injection | Phase 3 — LoRA |
|---|---|---|
| Knowledge location | Vector database (retrieved per query) | Model weights (always available) |
| Inference path | query → retrieve → inject KV → attention → answer | query → generate from weights → answer |
| Retrieval step | Required | Skipped for high-confidence queries |
| Context window usage | Consumes slots for injected KV | Zero — no context injected |
| Multi-hop reasoning | Limited to top-K chunks | Cross-chunk patterns encoded in weights |
| Staleness risk | Per-chunk, healed by background daemon | Per training round, one retrain |
| New data handling | Re-index the new chunk, KV computed automatically | Retrain needed to incorporate new knowledge |
| Approximate latency | ~1.5 s (inject + attention) | ~0.8 s (weights only, no retrieval) |
| When it applies | All queries | Queries where PRS confidence ≥ threshold |

#### When is Phase 2 the right stopping point?

Phase 3 fine-tuning makes sense when:

- The corpus is **stable and bounded** — a product manual, a legal document set, a knowledge base that updates infrequently
- Queries are **dense and repeated** — the same domain questions are asked over and over
- Retrieval quality is a **known bottleneck** — embedding space does not map well to the query vocabulary
- Latency budget is **tight** — sub-second answers from weights are required

Phase 2 alone is often the better operating point when:

- The corpus is **large and changing frequently** — continuous re-indexing is cheaper than periodic retraining
- Queries are **exploratory and diverse** — retrieval is more appropriate than parametric memory
- The team wants to **avoid the fine-tuning complexity** — Phase 2 is operationally simpler

#### The role of the A/B flywheel

The auto-curation flywheel directly addresses Phase 3's gap: when the cloud model (Model B) consistently produces a better answer than the fine-tuned local model (Model A) on a category of question, it signals a parametric knowledge gap — the LoRA has not learned that pattern. The curated Q&A pairs are appended to `faqs_curated.json` and used to retrain, patching exactly the chunks the model failed to internalize. This is the feedback loop that closes the gap between what Phase 2 retrieves correctly and what Phase 3 knows parametrically.

---

### Why store KV tensors in the vector database instead of a separate key-value store?

Storing KV tensors alongside the embeddings in the vector database is a deliberate co-location decision, not an architectural accident.

At query time, KVForge already performs a vector similarity search to retrieve the top-K chunks. The KV tensor for each chunk needs to be retrieved at the same moment — as part of the same payload read. If KV tensors were stored in a separate key-value store (Redis, S3, a file system), every query would require:

1. Vector search → get chunk IDs and embeddings
2. KV store lookup → fetch KV tensor for each chunk ID (additional round-trips)

Co-locating KV tensors in the vector database payload eliminates step 2 entirely. Retrieval and KV tensor loading happen in one network round-trip. At the latency targets KVForge is optimizing for (sub-second query response), eliminating those additional round-trips is meaningful.

The trade-off is that the vector database collection grows by ~115 KB per chunk (for Llama 3.2 3B). For 2,520 chunks that is ~290 MB of additional payload — well within Qdrant's operating range for a single collection.

---

### Why mean-pool the KV tensors over sequence length instead of storing per-token KV?

The full per-token KV tensor for a single chunk at shape `[num_layers, 2, num_kv_heads, seq_len, head_dim]` would be roughly `28 × 2 × 8 × 512 × 128 × 2 bytes ≈ 115 MB` per chunk (for Llama 3.2 3B with a 512-token chunk). At 2,520 chunks that is ~290 GB — clearly impossible to store in a vector database payload or load into GPU memory at query time.

Mean-pooling over the sequence dimension produces a fixed-size representation at `[num_layers, 2, num_kv_heads, head_dim]` — ~115 KB per chunk, a 1000× compression. The trade-off is that the pooled tensor is an approximation: it encodes the average attention state the model reached after processing the chunk, not the per-token granularity. This works well in practice because the retrieval system already ensures you are injecting the right chunk — you do not need per-token precision to get the semantic benefit of KV injection.

---

*Have a question not covered here? Open an issue at [github.com/hemantcgi/kvforge/issues](https://github.com/hemantcgi/kvforge/issues).*
