import pytest
from pydantic import BaseModel


def test_register_and_get_addon():
    from addons.registry import AddonRegistry, AddonManifest
    AddonRegistry.reset()

    class DummyConfig(BaseModel):
        port: int = 9000

    manifest = AddonManifest(
        name="dummy",
        display_name="Dummy Addon",
        description="Test addon.",
        config_schema=DummyConfig,
    )
    AddonRegistry.register(manifest)
    retrieved = AddonRegistry.get("dummy")
    assert retrieved.name == "dummy"
    assert retrieved.config_schema is DummyConfig


def test_register_duplicate_raises():
    from addons.registry import AddonRegistry, AddonManifest
    AddonRegistry.reset()

    class SchemaA(BaseModel):
        x: int = 1

    class SchemaB(BaseModel):
        y: str = "hello"

    m1 = AddonManifest(name="dup", display_name="D", description="D",
                       config_schema=SchemaA)
    m2 = AddonManifest(name="dup", display_name="D2", description="D2",
                       config_schema=SchemaB)
    AddonRegistry.register(m1)
    with pytest.raises(ValueError, match="already registered"):
        AddonRegistry.register(m2)


def test_get_unknown_raises():
    from addons.registry import AddonRegistry
    AddonRegistry.reset()
    with pytest.raises(KeyError, match="Unknown addon"):
        AddonRegistry.get("nonexistent")


def test_all_available_returns_list():
    from addons.registry import AddonRegistry, AddonManifest
    AddonRegistry.reset()

    class C1(BaseModel):
        pass

    class C2(BaseModel):
        pass

    AddonRegistry.register(AddonManifest(name="a", display_name="A",
                                          description="A", config_schema=C1))
    AddonRegistry.register(AddonManifest(name="b", display_name="B",
                                          description="B", config_schema=C2))
    names = [m.name for m in AddonRegistry.all_available()]
    assert "a" in names
    assert "b" in names


def test_validate_config_against_schema():
    from addons.registry import AddonRegistry, AddonManifest
    AddonRegistry.reset()

    class PortConfig(BaseModel):
        port: int = 8080
        debug: bool = False

    AddonRegistry.register(AddonManifest(name="webserver", display_name="Web",
                                          description="Web", config_schema=PortConfig))
    m = AddonRegistry.get("webserver")
    validated = m.config_schema(**{"port": 9999})
    assert validated.port == 9999
    assert validated.debug is False


def test_requires_field_defaults_empty():
    from addons.registry import AddonRegistry, AddonManifest
    AddonRegistry.reset()

    class C(BaseModel):
        pass

    m = AddonManifest(name="solo", display_name="S", description="S",
                      config_schema=C)
    assert m.requires == []


def test_load_builtins_registers_all_core_addons():
    from addons.registry import AddonRegistry
    AddonRegistry.reset()
    AddonRegistry.load_builtins()
    names = [m.name for m in AddonRegistry.all_available()]
    for expected in ["indexing", "inference", "training", "background",
                     "sync", "monitoring", "mcp", "model_scout",
                     "multimodal", "analytics"]:
        assert expected in names, f"Addon '{expected}' not registered after load_builtins()"
