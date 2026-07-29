"""Factory that instantiates the configured Embedder backend.

Provides a single ``get_embedder`` entry-point used by indexing and search
pipelines so that the rest of the codebase stays backend-agnostic.
"""


def get_embedder(cfg: dict):
    """Return the configured Embedder instance.

    Dispatches on ``cfg['embedder_backend']`` (default: ``'fastembed'``).

    Args:
        cfg: Datasource configuration dictionary.  Relevant keys:

            * ``embedder_backend`` — one of ``fastembed``,
              ``sentence_transformers``, ``openai``.
            * ``embed_model`` — model name or identifier passed to the backend
              (default ``'BAAI/bge-small-en-v1.5'``).
            * ``vector_dim`` — expected embedding dimensionality (default 384).
            * ``openai_api_key`` — API key for the OpenAI backend (falls back
              to the ``OPENAI_API_KEY`` environment variable).

    Returns:
        An ``Embedder``-protocol-compatible instance.

    Raises:
        ValueError: If ``cfg['embedder_backend']`` is not a recognised value.
    """
    # Support both flat configs and the nested addon_config.indexing layout.
    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    effective_cfg = {**cfg, **indexing_cfg}
    backend = effective_cfg.get("embedder_backend", "fastembed")
    model_name = effective_cfg.get("embed_model", "BAAI/bge-small-en-v1.5")
    dim = effective_cfg.get("vector_dim", 384)

    if backend == "fastembed":
        from embeddings.fastembed_embedder import FastEmbedEmbedder
        return FastEmbedEmbedder(model_name=model_name, dim=dim)
    if backend == "sentence_transformers":
        from embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model_name=model_name, dim=dim)
    if backend == "openai":
        from embeddings.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(model_name=model_name, dim=dim,
                               api_key=cfg.get("openai_api_key"))
    if backend == "clip":
        from embeddings.clip_embedder import CLIPEmbedder
        return CLIPEmbedder(model_name=cfg.get("clip_model", "openai/clip-vit-base-patch32"))
    raise ValueError(
        f"Unknown embedder_backend '{backend}'. Choose: fastembed, sentence_transformers, openai, clip"
    )
