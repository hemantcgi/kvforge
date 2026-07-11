from pydantic import BaseModel


class BackgroundConfig(BaseModel):
    """Configuration for the background KV recompute + access flush addon."""

    flush_seconds: int = 300   # flush access buffer every N seconds
    flush_queries: int = 50    # also flush after N queries (whichever comes first)
