from __future__ import annotations
from pydantic import BaseModel, Field


class MCPConfig(BaseModel):
    """Configuration for the MCP server addon (future implementation)."""

    port: int = 8765
    enabled_tools: list[str] = Field(
        default_factory=lambda: ["query", "status", "collections"]
    )
