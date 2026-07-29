"""EntiGraph-style entity-graph synthesis for factual-coverage diversity.

Extracts entities from each chunk, builds a corpus-level entity table with
cross-chunk connections, and generates relation texts for entity pairs that
co-occur in *different* chunks — producing factual diversity (new connections)
rather than expression diversity (paraphrases).

Output format: a JSON manifest compatible with ``lora_trainer.py``'s
dataset-manifest schema, so the trainer is unchanged.
"""
from __future__ import annotations

import json
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def _call_llm(prompt: str, provider: str, model: str, api_key: str) -> str:
    """Call a cloud LLM via the existing sleep_faq_generator._call_provider."""
    from pipeline.sleep_faq_generator import _call_provider
    return _call_provider(provider, model, api_key, prompt)


def _entity_extraction_prompt(chunk_text: str) -> str:
    return (
        "Extract all named entities from the following text as JSON.\n"
        "Format: {\"entities\": [{\"name\": str, \"type\": str, "
        "\"description\": str, \"facts\": [str]}]}\n\n"
        f"Text:\n{chunk_text}\n\n"
        "Output only the JSON, no preamble."
    )


def _relation_prompt(entity_a: dict, entity_b: dict, context_a: str, context_b: str) -> str:
    return (
        "Write a single factual sentence connecting these two entities.\n"
        "Base it only on the provided context.\n\n"
        f"Entity A: {entity_a['name']} — {entity_a['description']}\n"
        f"Context A: {context_a}\n\n"
        f"Entity B: {entity_b['name']} — {entity_b['description']}\n"
        f"Context B: {context_b}\n\n"
        "Output only the connecting sentence."
    )


def extract_entities(chunk_text: str, provider: str, model: str, api_key: str) -> list[dict]:
    """Extract named entities from a chunk via cloud LLM. Returns list of entity dicts."""
    prompt = _entity_extraction_prompt(chunk_text)
    raw = _call_llm(prompt, provider, model, api_key)
    try:
        data = json.loads(raw)
        return data.get("entities", [])
    except (json.JSONDecodeError, TypeError):
        return []


def build_entity_table(entities_by_chunk: dict[str, list[dict]]) -> dict:
    """Build a corpus-level entity table with cross-chunk connections.

    Args:
        entities_by_chunk: ``{chunk_id: [entity_dict, ...]}``.

    Returns:
        ``{entities: {name: {name, type, description, chunk_ids: [str]}},
           cross_chunk_pairs: [(name_a, name_b, chunk_a, chunk_b)]}``.
    """
    entities: dict[str, dict] = {}
    for chunk_id, ents in entities_by_chunk.items():
        for ent in ents:
            name = ent["name"]
            if name not in entities:
                entities[name] = {
                    "name": name,
                    "type": ent.get("type", "unknown"),
                    "description": ent.get("description", ""),
                    "chunk_ids": [],
                }
            if chunk_id not in entities[name]["chunk_ids"]:
                entities[name]["chunk_ids"].append(chunk_id)

    cross_chunk_pairs: list[tuple] = []
    entity_names_list = list(entities.keys())
    for i, name_a in enumerate(entity_names_list):
        for name_b in entity_names_list[i + 1:]:
            set_a = set(entities[name_a]["chunk_ids"])
            set_b = set(entities[name_b]["chunk_ids"])
            if set_a & set_b and set_a != set_b:
                diff_a = set_a - set_b
                diff_b = set_b - set_a
                shared = set_a & set_b
                if diff_a and diff_b:
                    for ca in diff_a:
                        for cb in diff_b:
                            cross_chunk_pairs.append((name_a, name_b, ca, cb))
                elif diff_a and not diff_b:
                    for ca in diff_a:
                        for cs in shared:
                            cross_chunk_pairs.append((name_a, name_b, ca, cs))
                elif not diff_a and diff_b:
                    for cs in shared:
                        for cb in diff_b:
                            cross_chunk_pairs.append((name_a, name_b, cs, cb))
    return {"entities": entities, "cross_chunk_pairs": cross_chunk_pairs}


def generate_relation_texts(
    entity_table: dict,
    chunks: dict[str, str],
    n_per_chunk: int,
    provider: str,
    model: str,
    api_key: str,
) -> list[dict]:
    """Generate relation texts for cross-chunk entity pairs.

    Args:
        entity_table: Output of ``build_entity_table``.
        chunks: ``{chunk_id: chunk_text}``.
        n_per_chunk: Target relation texts per chunk.

    Returns:
        List of ``{chunk_id, text, entity_pair: (name_a, name_b)}``.
    """
    entities = entity_table["entities"]
    pairs = entity_table["cross_chunk_pairs"]
    if not pairs:
        return []

    pairs_by_chunk: dict[str, list] = {}
    for name_a, name_b, chunk_a, chunk_b in pairs:
        pairs_by_chunk.setdefault(chunk_a, []).append((name_a, name_b, chunk_b))
        pairs_by_chunk.setdefault(chunk_b, []).append((name_b, name_a, chunk_a))

    results: list[dict] = []
    for chunk_id, chunk_pairs in pairs_by_chunk.items():
        sample = chunk_pairs[:n_per_chunk]
        for name_a, name_b, other_chunk in sample:
            ent_a = entities.get(name_a, {})
            ent_b = entities.get(name_b, {})
            context_a = chunks.get(chunk_id, "")
            context_b = chunks.get(other_chunk, "")
            prompt = _relation_prompt(ent_a, ent_b, context_a, context_b)
            text = _call_llm(prompt, provider, model, api_key).strip()
            if text:
                results.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "entity_pair": (name_a, name_b),
                })
    return results


def generate(
    chunks: list[dict],
    output_path: Path,
    n_per_chunk: int = 5,
    provider: str = "gemini",
    model: str = "gemini-2.5-flash",
    api_key: str = "",
) -> Path:
    """Full EntiGraph pipeline: extract → build table → generate relations → write manifest.

    Args:
        chunks: List of chunk dicts with ``chunk_id`` and ``text``.
        output_path: Where to write the JSON manifest.
        n_per_chunk: Target relation texts per chunk.
        provider: Cloud LLM provider.
        model: Cloud LLM model.
        api_key: API key.

    Returns:
        The output Path (for chaining).
    """
    entities_by_chunk: dict[str, list[dict]] = {}
    for c in chunks:
        cid = str(c["chunk_id"])
        entities_by_chunk[cid] = extract_entities(c["text"], provider, model, api_key)

    table = build_entity_table(entities_by_chunk)

    chunk_texts = {str(c["chunk_id"]): c["text"] for c in chunks}
    relations = generate_relation_texts(table, chunk_texts, n_per_chunk,
                                        provider, model, api_key)

    manifest = {
        "training_examples": [
            {"chunk_id": r["chunk_id"], "text": r["text"], "type": "relation"}
            for r in relations
        ],
        "entity_count": len(table["entities"]),
        "cross_chunk_pair_count": len(table["cross_chunk_pairs"]),
        "n_per_chunk": n_per_chunk,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    entities_path = output_path.parent / "entigraph_entities.json"
    entities_payload = {
        name: {"chunk_ids": ent["chunk_ids"], "type": ent.get("type", "")}
        for name, ent in table["entities"].items()
    }
    entities_path.write_text(json.dumps(entities_payload, indent=2, ensure_ascii=False))
    return output_path
