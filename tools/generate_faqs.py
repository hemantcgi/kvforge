"""tools/generate_faqs.py — Auto-generate FAQs from an indexed corpus.

Usage:
  python tools/generate_faqs.py --config datasource_example.json --count 50 --output faqs.json

Algorithm:
  1. Sample N chunks from the vector store collection
  2. For each chunk, prompt the LLM: "Generate one factual Q&A pair..."
  3. Parse Q and A from the LLM output
  4. Save to output JSON file
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


QA_PROMPT_TEMPLATE = """Read the following passage and generate exactly one factual question and answer pair about it.

Passage:
{chunk}

Respond in this exact format:
Q: <question>
A: <answer>"""


def _parse_qa(text: str):
    """Parse a Q&A pair from LLM output. Returns (question, answer) or None."""
    q_match = re.search(r"(?:Q|Question):\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    a_match = re.search(r"(?:A|Answer):\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if not q_match or not a_match:
        return None
    q = q_match.group(1).strip()
    a = a_match.group(1).strip()
    if not q or not a:
        return None
    return q, a


def _sample_chunks(store, collection: str, n: int) -> list[str]:
    """Sample up to n chunk texts from the vector store."""
    results, _ = store.scroll(collection, limit=500, with_payload=True, with_vectors=False)
    texts = [r.payload.get("text", "") for r in results if r.payload.get("text")]
    return random.sample(texts, min(n, len(texts)))


def generate(cfg: dict, count: int, output_path: str) -> None:
    """Main generation loop: sample chunks, prompt LLM, save FAQs."""
    import model_loader

    model_loader.init(cfg)
    model, tokenizer = model_loader.load()

    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=128, do_sample=False)

    # Use VectorStore abstraction if available, else fall back to direct qdrant
    try:
        from vectorstore.registry import get_store
        store = get_store(cfg)
    except ImportError:
        from qdrant_client import QdrantClient
        store = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])

    sampled = _sample_chunks(store, cfg["collection"], n=count * 2)

    q_key = cfg.get("faq_question_key", "question")
    a_key = cfg.get("faq_answer_key", "answer")
    faqs = []

    for chunk in sampled:
        if len(faqs) >= count:
            break
        prompt = QA_PROMPT_TEMPLATE.format(chunk=chunk[:1000])
        out = pipe(prompt)[0]["generated_text"][len(prompt):].strip()
        parsed = _parse_qa(out)
        if parsed is None:
            continue
        q, a = parsed
        faqs.append({q_key: q, a_key: a})
        print(f"  [{len(faqs)}/{count}] Q: {q[:80]}")

    with open(output_path, "w") as f:
        json.dump(faqs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(faqs)} FAQs to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-generate FAQs from an indexed corpus")
    p.add_argument("--config", required=True, help="Datasource config JSON")
    p.add_argument("--count", type=int, default=50, help="Number of FAQs to generate")
    p.add_argument("--output", default="generated_faqs.json", help="Output JSON path")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    generate(cfg, args.count, args.output)


if __name__ == "__main__":
    main()
