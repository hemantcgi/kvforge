"""
ModelScout CLI — interactive model selection agent.

Usage:
  python -m pipeline.model_scout_cli --config my_config.json --docs ./docs/
  python -m pipeline.model_scout_cli --config my_config.json --faqs faqs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelScout: interactive model selector")
    parser.add_argument("--config", required=True, help="Path to UC config JSON")
    parser.add_argument("--docs", default=None, help="Document directory (pre-index mode)")
    parser.add_argument("--faqs", default=None, help="Existing FAQ JSON file (optional)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    from pipeline.model_scout import CLIAdapter, detect_gpu, run_scout_session

    adapter = CLIAdapter()
    gpu_info = detect_gpu()

    # Build FAQ set
    if args.faqs and Path(args.faqs).exists():
        with open(args.faqs) as f:
            faqs = json.load(f)
        adapter.send(f"Loaded {len(faqs)} FAQs from {args.faqs}")
    else:
        faq_count = cfg.get("scout_initial_faq_count", 20)
        adapter.send(f"Generating {faq_count} FAQs from corpus...")
        try:
            from pipeline.sleep_faq_generator import generate as generate_faqs
            faqs = generate_faqs(cfg, count=faq_count, source_dir=args.docs)
        except Exception as exc:
            adapter.send(f"FAQ generation failed: {exc}. Using empty FAQ set.")
            faqs = []
        adapter.send(f"Generated {len(faqs)} FAQs.")

    # Attempt to connect to vector store (post-index mode)
    store = None
    try:
        from vectorstore.registry import get_store
        store = get_store(cfg)
    except Exception:
        adapter.send("Could not connect to vector store — running in pre-index mode.")

    recommendation = run_scout_session(adapter, cfg, faqs, gpu_info, store=store)

    if recommendation:
        adapter.send(
            f"\nTo use this model, set in {args.config}:\n"
            f'  "llm_model": "{recommendation["model_id"]}",\n'
            f'  "quantization": "{recommendation["quantization"]}",\n'
            f'  "lora_rank": {recommendation["lora_rank"]}'
        )
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
