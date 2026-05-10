"""Detect archived chunks that have crossed the reinstatement frequency threshold."""


def check_reinstatement_candidates(
    archived_chunks: list[dict],
    threshold: int,
) -> list[dict]:
    """Return archived chunks whose archive_retrieval_count >= threshold."""
    return [
        c for c in archived_chunks
        if c.get("payload", c).get("archive_retrieval_count", 0) >= threshold
    ]


def build_reinstatement_recommendation(chunk: dict, threshold: int) -> dict:
    """Build a dashboard recommendation payload for a reinstatement candidate."""
    payload = chunk.get("payload", chunk)
    count   = payload.get("archive_retrieval_count", 0)
    text_preview = (payload.get("text", "") or "")[:120].replace("\n", " ")
    return {
        "chunk_id":        chunk["id"],
        "action":          "reinstate",
        "reason":          (f"Retrieved from archive {count} times "
                            f"(threshold: {threshold}). "
                            f"Content may be more important than originally assessed."),
        "text_preview":    text_preview,
        "retrieval_count": count,
    }
