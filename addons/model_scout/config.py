from pydantic import BaseModel


class ModelScoutConfig(BaseModel):
    """Configuration for the ModelScout evaluation addon."""

    model_registry_path: str = "core/model_registry.json"
    model_scout_program: str = "model_scout_program.md"
    model_scout_results: str = "model_scout_results.tsv"
    initial_corpus_chunks: int = 200
    initial_faq_count: int = 20
    initial_lora_steps: int = 500
    initial_lora_rank: int = 16
    max_lora_steps: int = 2000
    max_corpus_chunks: int = 2000
    max_faq_count: int = 100
