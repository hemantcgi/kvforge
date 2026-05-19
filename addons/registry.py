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

        Idempotent: re-registering the same name with the same config_schema is a no-op
        (safe for module reloads). Raises ValueError if a *different* config_schema
        tries to claim the same name.
        """
        if manifest.name in cls._manifests:
            existing = cls._manifests[manifest.name]
            if existing.config_schema is manifest.config_schema:
                return  # same schema — idempotent re-registration
            # Allow update when the schema class was recreated by importlib.reload()
            # (same qualified name + module means same addon, different class object)
            same_class = (
                existing.config_schema.__qualname__ == manifest.config_schema.__qualname__
                and existing.config_schema.__module__ == manifest.config_schema.__module__
            )
            if same_class:
                cls._manifests[manifest.name] = manifest
                return
            raise ValueError(
                f"Addon '{manifest.name}' already registered with a different config schema."
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
        """Import (or reload) all built-in addon packages so they self-register.

        Uses importlib.reload() on already-cached modules so that test code calling
        reset() + load_builtins() gets fresh registration even within a single process.
        """
        import importlib
        import sys

        _builtin_modules = [
            "addons.indexing",
            "addons.inference",
            "addons.training",
            "addons.background",
            "addons.sync",
            "addons.monitoring",
            "addons.mcp",
            "addons.model_scout",
            "addons.multimodal",
            "addons.analytics",
            "addons.turboquant",
            "addons.corpus_intelligence",
            "addons.compute",
        ]
        for mod_name in _builtin_modules:
            try:
                if mod_name in sys.modules:
                    importlib.reload(sys.modules[mod_name])
                else:
                    importlib.import_module(mod_name)
            except ModuleNotFoundError:
                pass  # addon package not installed yet — skip silently

    @classmethod
    def reset(cls) -> None:
        """Clear all registered addons. For use in tests only."""
        cls._manifests.clear()
