from pydantic import BaseModel


class AnalyticsConfig(BaseModel):
    """Configuration for the flywheel analytics addon."""

    analytics_db: str = ""
    cost_per_1k_tokens: float = 5.0
    tokens_per_ms_baseline: float = 0.8
