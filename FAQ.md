# SmartQdrant — Frequently Asked Questions

This index links to per-section detail pages. Each section page contains all question entries and full answers for that topic.

---

## Table of Contents

**Vector Stores** → [View full section](docs/faq/vector-stores.md)
- [How do I use SmartQdrant with ChromaDB instead of Qdrant?](docs/faq/vector-stores.md#how-do-i-use-smartqdrant-with-chromadb-instead-of-qdrant)
- [How do I add support for Pinecone, Weaviate, or another vector database?](docs/faq/vector-stores.md#how-do-i-add-support-for-pinecone-weaviate-or-another-vector-database)
- [How do I use SmartQdrant with pgvector (PostgreSQL)?](docs/faq/vector-stores.md#how-do-i-use-smartqdrant-with-pgvector-postgresql)
- [How do I use SmartQdrant with FAISS?](docs/faq/vector-stores.md#how-do-i-use-smartqdrant-with-faiss)
- [How do I use SmartQdrant with Milvus or Zilliz Cloud?](docs/faq/vector-stores.md#how-do-i-use-smartqdrant-with-milvus-or-zilliz-cloud)
- [How do I use SmartQdrant with LanceDB?](docs/faq/vector-stores.md#how-do-i-use-smartqdrant-with-lancedb)
- [How do I use SmartQdrant with Redis (RedisSearch)?](docs/faq/vector-stores.md#how-do-i-use-smartqdrant-with-redis-redissearch)
- [How do I use SmartQdrant with Elasticsearch or OpenSearch?](docs/faq/vector-stores.md#how-do-i-use-smartqdrant-with-elasticsearch-or-opensearch)
- [How do I use SmartQdrant with MongoDB Atlas Vector Search?](docs/faq/vector-stores.md#how-do-i-use-smartqdrant-with-mongodb-atlas-vector-search)
- [Can I use an existing Qdrant collection I already have?](docs/faq/vector-stores.md#can-i-use-an-existing-qdrant-collection-i-already-have)

**Language Models** → [View full section](docs/faq/language-models.md)
- [How do I use my own LLM for KV computation?](docs/faq/language-models.md#how-do-i-use-my-own-llm-for-kv-computation)
- [How do I use a gated model like Llama 3 that requires a HuggingFace token?](docs/faq/language-models.md#how-do-i-use-a-gated-model-like-llama-3-that-requires-a-huggingface-token)
- [Can I use an API-hosted LLM (OpenAI, Anthropic, Gemini)?](docs/faq/language-models.md#can-i-use-an-api-hosted-llm-openai-anthropic-gemini)
- [Can I run this without a GPU?](docs/faq/language-models.md#can-i-run-this-without-a-gpu)

**Embedding Models** → [View full section](docs/faq/embedding-models.md)
- [How do I use OpenAI embeddings instead of FastEmbed?](docs/faq/embedding-models.md#how-do-i-use-openai-embeddings-instead-of-fastembed)
- [How do I use sentence-transformers embeddings?](docs/faq/embedding-models.md#how-do-i-use-sentence-transformers-embeddings)
- [How do I add a custom embedding model?](docs/faq/embedding-models.md#how-do-i-add-a-custom-embedding-model)
- [Can I use different embedding models for different collections?](docs/faq/embedding-models.md#can-i-use-different-embedding-models-for-different-collections)

**Document Ingestion** → [View full section](docs/faq/document-ingestion.md)
- [How do I index Markdown documentation?](docs/faq/document-ingestion.md#how-do-i-index-markdown-documentation)
- [How do I index a JSONL dataset?](docs/faq/document-ingestion.md#how-do-i-index-a-jsonl-dataset)
- [How do I index HTML pages or web content?](docs/faq/document-ingestion.md#how-do-i-index-html-pages-or-web-content)
- [How do I index an entire directory of mixed file types?](docs/faq/document-ingestion.md#how-do-i-index-an-entire-directory-of-mixed-file-types)
- [How do I add support for a custom document format?](docs/faq/document-ingestion.md#how-do-i-add-support-for-a-custom-document-format)

**KV Cache & Phases** → [View full section](docs/faq/kv-cache-and-phases.md)
- [What exactly is stored in the KV cache payload?](docs/faq/kv-cache-and-phases.md#what-exactly-is-stored-in-the-kv-cache-payload)
- [How does KV injection work under the hood?](docs/faq/kv-cache-and-phases.md#how-does-kv-injection-work-under-the-hood)
- [Why do some queries fall back to text-in-context even in Phase 2?](docs/faq/kv-cache-and-phases.md#why-do-some-queries-fall-back-to-text-in-context-even-in-phase-2)
- [How do I manually advance or roll back the phase?](docs/faq/kv-cache-and-phases.md#how-do-i-manually-advance-or-roll-back-the-phase)

**Training & PRS** → [View full section](docs/faq/training-and-prs.md)
- [How do I tune the PRS threshold?](docs/faq/training-and-prs.md#how-do-i-tune-the-prs-threshold)
- [My PRS is not improving across training rounds — what do I do?](docs/faq/training-and-prs.md#my-prs-is-not-improving-across-training-rounds--what-do-i-do)
- [How do I bring my own FAQs for PRS evaluation?](docs/faq/training-and-prs.md#how-do-i-bring-my-own-faqs-for-prs-evaluation)
- [How do I change the PRS scoring weights?](docs/faq/training-and-prs.md#how-do-i-change-the-prs-scoring-weights)

**Multi-Corpus & Production** → [View full section](docs/faq/multi-corpus-and-production.md)
- [Can I run multiple independent corpora on the same instance?](docs/faq/multi-corpus-and-production.md#can-i-run-multiple-independent-corpora-on-the-same-instance)
- [How do I keep KV tensors fresh when I update my documents?](docs/faq/multi-corpus-and-production.md#how-do-i-keep-kv-tensors-fresh-when-i-update-my-documents)
- [How do I monitor what is happening at runtime?](docs/faq/multi-corpus-and-production.md#how-do-i-monitor-what-is-happening-at-runtime)
- [How do I reset everything and start over?](docs/faq/multi-corpus-and-production.md#how-do-i-reset-everything-and-start-over)
- [What are the GPU memory requirements?](docs/faq/multi-corpus-and-production.md#what-are-the-gpu-memory-requirements)

---

*Have a question not covered here? Open an issue at [github.com/hemantcgi/smartqdrant/issues](https://github.com/hemantcgi/smartqdrant/issues).*
