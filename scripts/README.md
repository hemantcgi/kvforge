# KVForge Scripts

Shell wrappers for every KVForge Python tool. All scripts:
- Validate that Python is available
- Validate required arguments before running
- Use `set -euo pipefail` (fail fast on errors)
- Print usage if called with missing arguments

## Quick Reference

| Script | Purpose | Required args |
|--------|---------|---------------|
| `ask.sh` | Query KVForge | `config.json` `"query"` |
| `index.sh` | Index documents into vector store | `config.json` `source_path` |
| `compute_kv.sh` | Compute KV tensors (Phase 1→2) | `config.json` |
| `train_lora.sh` | LoRA fine-tuning | `config.json` `faqs.json` |
| `evaluate_prs.sh` | Evaluate Parametric Readiness Score | `config.json` `faqs.json` |
| `generate_faqs.sh` | Auto-generate FAQ pairs from corpus | `config.json` `output.json` |
| `dashboard.sh` | Start monitoring dashboard | `config.json` |
| `generate_docs.sh` | Generate HTML API docs from docstrings | _(none)_ |
| `run_pipeline.sh` | Full Phase 1→2→3 pipeline | `config.json` `source` `faqs.json` |

## Common Workflows

### Index and query (no GPU needed)

```bash
python kvforge.py init --name my-corpus
./scripts/index.sh datasource_my-corpus.json ./my-docs/
./scripts/ask.sh datasource_my-corpus.json "What is my question?"
```

### Full pipeline (GPU required for KV + LoRA)

```bash
./scripts/run_pipeline.sh datasource_my-corpus.json ./my-docs/ my-corpus_faqs.json
```

### Individual steps

```bash
# Generate FAQs from indexed corpus
./scripts/generate_faqs.sh datasource_my-corpus.json my-corpus_faqs.json 50

# Fine-tune with existing FAQs
./scripts/train_lora.sh datasource_my-corpus.json my-corpus_faqs.json

# Evaluate readiness
./scripts/evaluate_prs.sh datasource_my-corpus.json my-corpus_faqs.json

# Start monitoring
./scripts/dashboard.sh datasource_my-corpus.json 8080
```

## Each script in detail

Each script contains a full header comment block with:
- Description
- Usage syntax
- Required and optional arguments
- Prerequisites (GPU, services)
- Example invocation

Run any script without arguments to see its usage.
