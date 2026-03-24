"""embeddings/registry.py — Factory for Embedder implementations."""


def get_embedder(cfg: dict):
    """Return the configured Embedder instance.

    Dispatches on cfg['embedder_backend'] (default: 'fastembed').
    """
    backend = cfg.get("embedder_backend", "fastembed")
    model_name = cfg.get("embed_model", "BAAI/bge-small-en-v1.5")
    dim = cfg.get("vector_dim", 384)

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
    raise ValueError(
        f"Unknown embedder_backend '{backend}'. Choose: fastembed, sentence_transformers, openai"
    )
