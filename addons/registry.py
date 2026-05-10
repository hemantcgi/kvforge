"""AddonManifest dataclass and AddonRegistry singleton for KVForge addons."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel

if TYPE_CHECKING:
    from fastapi import APIRouter


@dataclass
class AddonManifest:
    """Describes a KVForge addon: its config schema, lifecycle hooks, and Dashboard routes.

    Attributes:
        name: Unique identifier used in KVForgeConfig.addons and addon_config keys.
        display_name: Human-readable name shown in the Dashboard setup wizard.
        description: One-sentence description shown in the Dashboard.
        config_schema: Pydantic BaseModel subclass that validates addon_config[name].
        requires: Names of other addons that must be active for this addon to work.
        startup: Optional callable invoked with the raw addon config dict at startup.
        shutdown: Optional callable invoked with no arguments at shutdown.
        dashboard_routes: Optional callable returning an APIRouter of Dashboard routes.
    """

    name: str
    display_name: str
    description: str
    config_schema: type[BaseModel]
    requires: list[str] = field(default_factory=list)
    startup: Callable[[dict], None] | None = None
    shutdown: Callable[[], None] | None = None
    dashboard_routes: Callable[[], "APIRouter"] | None = None


class AddonRegistry:
    """Singleton registry of available KVForge addons.

    Addons self-register at import time via their __init__.py.
    Call load_builtins() at application startup to import all built-in addons.
    """

    _manifests: dict[str, AddonManifest] = {}

    @classmethod
    def register(cls, manifest: AddonManifest) -> None:
        """Register an addon manifest.

        Raises:
            ValueError: If an addon with the same name is already registered.
        """
        if manifest.name in cls._manifests:
            raise ValueError(
                f"Addon '{manifest.name}' already registered. "
                f"Each addon name must be unique."
            )
        cls._manifests[manifest.name] = manifest

    @classmethod
    def get(cls, name: str) -> AddonManifest:
        """Return the manifest for a registered addon.

        Raises:
            KeyError: If no addon with that name is registered.
        """
        if name not in cls._manifests:
            raise KeyError(
                f"Unknown addon '{name}'. "
                f"Available: {sorted(cls._manifests.keys())}"
            )
        return cls._manifests[name]

    @classmethod
    def all_available(cls) -> list[AddonManifest]:
        """Return all registered addon manifests in registration order."""
        return list(cls._manifests.values())

    @classmethod
    def load_builtins(cls) -> None:
        """Import all built-in addon packages so they self-register.

        Safe to call multiple times — addons guard against duplicate registration.
        """
        import addons.indexing    # noqa: F401
        import addons.inference   # noqa: F401
        import addons.training    # noqa: F401
        import addons.background  # noqa: F401
        import addons.sync        # noqa: F401
        import addons.monitoring  # noqa: F401
        import addons.mcp         # noqa: F401
        import addons.model_scout # noqa: F401
        import addons.multimodal  # noqa: F401
        import addons.analytics   # noqa: F401

    @classmethod
    def reset(cls) -> None:
        """Clear all registered addons. For use in tests only."""
        cls._manifests.clear()
