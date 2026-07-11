from addons.registry import AddonRegistry, AddonManifest
from addons.indexing.config import IndexingConfig

AddonRegistry.register(AddonManifest(
    name="indexing",
    display_name="KV Indexer",
    description=(
        "Chunk documents, compute embeddings and KV tensors, "
        "upsert to the configured vector store."
    ),
    config_schema=IndexingConfig,
    requires=[],
))
