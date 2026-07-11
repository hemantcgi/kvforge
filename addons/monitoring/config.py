from pydantic import BaseModel


class MonitoringConfig(BaseModel):
    """Configuration for the per-UC monitoring dashboard addon."""

    port: int = 8082
    analytics_db: str = ""
    cost_per_1k_tokens: float = 5.0
    tokens_per_ms_baseline: float = 0.8
