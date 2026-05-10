from addons.registry import AddonRegistry, AddonManifest
from addons.mcp.config import MCPConfig

AddonRegistry.register(AddonManifest(
    name="mcp",
    display_name="MCP Server",
    description=(
        "Exposes KVForge as an MCP-compatible tool server for Claude, "
        "LangGraph, AutoGen, and any MCP-aware agent framework."
    ),
    config_schema=MCPConfig,
    requires=["inference"],
))
