"""Archival and reinstatement execution (called on user confirmation)."""
from pathlib import Path


def archive_chunk(chunk: dict, backend, vector_store, collection: str) -> None:
    """Execute archival for one chunk (user has confirmed via dashboard).

    Steps:
      1. Write chunk text to archival backend.
      2. Delete per-token KV file from disk (if present).
      3. Update vector store: clear kv_cache, set status=archived, store pointer.
    """
    payload  = chunk.get("payload", chunk)
    chunk_id = chunk["id"]
    text     = payload.get("text", "")
    kv_path  = payload.get("kv_token_path")

    backend.write(chunk_id, text)
    pointer = backend.get_pointer(chunk_id)

    if kv_path:
        p = Path(kv_path)
        if p.exists():
            p.unlink()

    vector_store.update_payload(
        collection=collection,
        point_id=chunk_id,
        payload={
            "kv_cache": "",
            "kv_token_path": None,
            "status": "archived",
            "archive_path": pointer,
            "archive_retrieval_count": 0,
        },
    )


def reinstate_chunk(chunk: dict, backend, vector_store, collection: str,
                    model, tokenizer, cfg: dict) -> None:
    """Execute reinstatement (user confirmed via dashboard).

    Steps:
      1. Fetch text from archival backend.
      2. Run LLM forward pass → mean_pool_kv → write kv_cache back.
      3. Update vector store: restore status, clear archive fields.
      4. Delete text from archival backend.
    """
    from core.kv_utils import mean_pool_kv, serialize_kv
    import torch

    payload  = chunk.get("payload", chunk)
    chunk_id = chunk["id"]
    text     = backend.read(chunk_id) or payload.get("text", "")

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=cfg.get("chunk_size", 512))
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    kv_arr = mean_pool_kv(out.past_key_values)
    kv_b64 = serialize_kv(kv_arr)

    vector_store.update_payload(
        collection=collection,
        point_id=chunk_id,
        payload={
            "kv_cache": kv_b64,
            "kv_token_path": None,
            "status": "active",
            "archive_path": None,
            "archive_retrieval_count": 0,
        },
    )
    backend.delete(chunk_id)
