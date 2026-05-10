"""Lean core configuration model for a single KVForge use case.

All component-specific configuration lives in addon_config, keyed by addon name.
The full configuration for a pipeline component is produced by get_merged_config().

Example JSON format::

    {
      "use_case_name": "Customer Support RAG",
      "collection": "customer-support",
      "version_file": "examples/usecase1/version.json",
      "addons": ["indexing", "inference", "training", "background", "sync", "monitoring"],
      "addon_config": {
        "indexing": {
          "loader": "jsonl",
          "embed_model": "BAAI/bge-small-en-v1.5",
          "vector_dim": 384,
          "vector_store": "qdrant",
          "qdrant_host": "localhost",
          "qdrant_port": 6333
        },
        "inference": {
          "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
          "top_k": 5,
          "gate_threshold": 0.75
        },
        "training": {
          "lora_rank": 16,
          "checkpoint_dir": "examples/usecase1/lora_checkpoints/",
          "replay_db": "examples/usecase1/replay.db"
        },
        "background": {"flush_seconds": 300, "flush_queries": 50},
        "sync": {"interval_minutes": 60},
        "monitoring": {"port": 8081}
      }
    }
"""
from __future__ import annotations

import json
from pydantic import BaseModel, Field


class KVForgeConfig(BaseModel):
    """Lean core configuration for one KVForge use case.

    Contains only the five fields that are truly universal. Everything else
    lives inside addon_config, validated lazily by each addon's own schema
    when the addon is activated.

    Attributes:
        use_case_name: Human-readable display name for this use case.
        collection: Vector store collection name. Shared across all addons.
        version_file: Path to the JSON file tracking LoRA version and phase.
            Shared by inference, training, and background addons.
        addons: List of addon names to activate. Order does not matter.
            Each name must match a registered AddonManifest.name.
        addon_config: Per-addon configuration dicts, keyed by addon name.
            Each dict is validated by the corresponding addon's config_schema
            when get_validated_addon_config() is called.
    """

    use_case_name: str
    collection: str
    version_file: str
    addons: list[str] = Field(default_factory=list)
    addon_config: dict[str, dict] = Field(default_factory=dict)

    def has_addon(self, name: str) -> bool:
        """Return True if ``name`` is in the active addons list."""
        return name in self.addons

    def get_merged_config(self, *addon_names: str) -> dict:
        """Merge core fields + one or more addon configs into a flat dict.

        This is the bridge between the new nested config format and the
        existing pipeline code, which accepts plain dicts via cfg.get().
        Core fields (collection, version_file, use_case_name) are always
        included. Addon configs are merged in the order given; later addons
        override earlier ones on key conflict.

        Args:
            *addon_names: Names of addons whose config_dicts to merge.
                Unknown addon names produce no keys (no KeyError).

        Returns:
            Flat dict suitable for passing directly to pipeline functions.

        Example::

            merged = cfg.get_merged_config("inference", "training")
            # merged["collection"] == cfg.collection
            # merged["llm_model"] == cfg.addon_config["inference"]["llm_model"]
            # merged["lora_rank"] == cfg.addon_config["training"]["lora_rank"]
        """
        result: dict = {
            "collection": self.collection,
            "version_file": self.version_file,
            "use_case_name": self.use_case_name,
        }
        for name in addon_names:
            result.update(self.addon_config.get(name, {}))
        return result

    def get_validated_addon_config(self, addon_name: str):
        """Return the addon's config as its typed Pydantic model.

        Looks up the registered AddonManifest for addon_name, then validates
        addon_config[addon_name] against the manifest's config_schema.

        Args:
            addon_name: Must be registered in AddonRegistry.

        Returns:
            Instance of the addon's config_schema Pydantic model.

        Raises:
            KeyError: If addon_name is not registered.
            pydantic.ValidationError: If addon_config[addon_name] is invalid.
        """
        from addons.registry import AddonRegistry
        manifest = AddonRegistry.get(addon_name)
        raw = self.addon_config.get(addon_name, {})
        return manifest.config_schema(**raw)

    def validate_addon_deps(self) -> None:
        """Check that all addon dependency requirements are satisfied.

        For every addon in self.addons, looks up its AddonManifest and
        verifies that each name in manifest.requires is also present in
        self.addons.

        Raises:
            ValueError: First unsatisfied dependency found, with a clear message.
            KeyError: If an addon name is not registered in AddonRegistry.
        """
        from addons.registry import AddonRegistry
        active = set(self.addons)
        for name in self.addons:
            manifest = AddonRegistry.get(name)
            for req in manifest.requires:
                if req not in active:
                    raise ValueError(
                        f"Addon '{name}' requires addon '{req}', "
                        f"but '{req}' is not in the active addons list. "
                        f"Add '{req}' to the addons list in your config."
                    )


def load_config(path: str) -> KVForgeConfig:
    """Load and validate a KVForgeConfig from a JSON file.

    Keys starting with ``_`` (e.g. ``_comment``) are stripped before
    validation so annotated template files can be used directly.

    Args:
        path: Path to the JSON config file.

    Returns:
        Validated KVForgeConfig instance.

    Raises:
        pydantic.ValidationError: If required fields are missing or invalid.
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path) as f:
        data = json.load(f)
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    return KVForgeConfig(**data)
