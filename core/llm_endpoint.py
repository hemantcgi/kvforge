"""Centralized LLM endpoint configuration and client factory.

Supports OpenAI-compatible endpoints, Google Gemini, and Anthropic Claude.
API keys support ``${ENV_VAR}`` substitution and direct string values.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Literal

Provider = Literal["openai", "gemini", "claude"]

_ENV_VAR_RE = re.compile(r"^\$\{(\w+)\}$")


def _resolve_key(raw: str | None) -> str | None:
    if raw is None:
        return None
    m = _ENV_VAR_RE.match(raw)
    if m:
        return os.environ.get(m.group(1))
    return raw


@dataclass
class LlmEndpointConfig:
    """Configuration for a single LLM endpoint.

    Attributes match the keys in ``llm_endpoints.json`` or ``uc_config.json``'s
    ``llm.*`` block.  Every field has a sensible default so partial configs work.
    """

    provider: Provider = "openai"
    """Which provider SDK to use.  ``"openai"`` also covers any OpenAI-compatible
    server (Fireworks, vLLM, Together, etc.) when *base_url* is set."""

    base_url: str = ""
    """Base URL for the API.  Leave empty to use the SDK's default
    (``https://api.openai.com/v1`` for OpenAI, etc.).  Ignored for Gemini."""

    api_key: str | None = None
    """API key.  Supports ``${ENV_VAR}`` syntax.  Falls back to the provider's
    standard environment variable if ``None`` after resolution."""

    model: str = ""
    """Model name (e.g. ``gpt-4o-mini``, ``gemini-2.5-flash``)."""

    temperature: float = 0.0
    """Sampling temperature."""

    max_tokens: int = 1024
    """Maximum tokens in the response."""

    # ------------------------------------------------------------------
    # Backing env-var names known by the broader codebase
    _ENV_KEYS: dict[str, str] = field(default_factory=lambda: {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    })

    def resolve_api_key(self) -> str | None:
        key = _resolve_key(self.api_key)
        if key:
            return key
        env_name = self._ENV_KEYS.get(self.provider)
        if env_name:
            return os.environ.get(env_name)
        return None


def _load_from_uc_config(uc_path: str, prefix: str = "sleep_faq") -> LlmEndpointConfig:
    """Read an endpoint config block from a ``uc_config.json`` file.

    Args:
        uc_path: Path to ``uc_config.json``.
        prefix: Key prefix inside ``llm``, e.g. ``"sleep_faq"`` reads
            ``llm.sleep_faq_provider``, ``llm.sleep_faq_model``, etc.

    Returns:
        Populated endpoint config (fields missing in the file keep defaults).
    """
    import json
    try:
        uc = json.load(open(uc_path))
    except (FileNotFoundError, json.JSONDecodeError):
        return LlmEndpointConfig()

    llm_block = uc.get("llm", {})
    return LlmEndpointConfig(
        provider=llm_block.get(f"{prefix}_provider", "openai"),
        model=llm_block.get(f"{prefix}_model", ""),
        base_url=llm_block.get(f"{prefix}_base_url", ""),
        api_key=llm_block.get(f"{prefix}_api_key", None),
        temperature=llm_block.get(f"{prefix}_temperature", 0.0),
        max_tokens=llm_block.get(f"{prefix}_max_tokens", 1024),
    )


def _load_from_json(path: str, endpoint_name: str) -> LlmEndpointConfig:
    """Read a named endpoint from a standalone ``llm_endpoints.json`` file.

    The file format::

        {
          "endpoints": {
            "faq_gen": { "provider": "openai", "base_url": "...", ... },
            "judge":   { ... },
            "inference": { ... }
          }
        }
    """
    import json
    try:
        data = json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return LlmEndpointConfig()

    ep = data.get("endpoints", {}).get(endpoint_name, {})
    return LlmEndpointConfig(
        provider=ep.get("provider", "openai"),
        model=ep.get("model", ""),
        base_url=ep.get("base_url", ""),
        api_key=ep.get("api_key", None),
        temperature=ep.get("temperature", 0.0),
        max_tokens=ep.get("max_tokens", 1024),
    )


def make_client(cfg: LlmEndpointConfig):
    """Build a client object for the configured endpoint.

    Returns a tuple ``(client, resolved_cfg)`` where *client* is one of:

    * ``openai.OpenAI`` (or an OpenAI-compatible wrapper)
    * ``anthropic.Anthropic``
    * ``google.generativeai`` module reference (for Gemini)

    The *resolved_cfg* is a new ``LlmEndpointConfig`` with the API key
    resolved (env-var substitution done) so callers can read the final
    values without repeating resolution logic.

    Usage::

        client, cfg = make_client(endpoint_cfg)
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    """
    import copy
    resolved = copy.deepcopy(cfg)
    resolved.api_key = cfg.resolve_api_key()
    api_key = resolved.api_key

    if cfg.provider == "gemini":
        import google.generativeai as genai
        if api_key:
            genai.configure(api_key=api_key)
        return genai, resolved

    if cfg.provider == "claude":
        import anthropic
        kwargs = {"api_key": api_key} if api_key else {}
        return anthropic.Anthropic(**kwargs), resolved

    # --- OpenAI / OpenAI-compatible ---
    import openai
    kwargs: dict = {}
    if api_key:
        kwargs["api_key"] = api_key
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url.rstrip("/") + "/"
    return openai.OpenAI(**kwargs), resolved
