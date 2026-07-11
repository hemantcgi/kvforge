from addons.registry import AddonRegistry, AddonManifest
from addons.corpus_intelligence.config import CorpusIntelligenceConfig

AddonRegistry.register(AddonManifest(
    name="corpus_intelligence",
    display_name="Corpus Intelligence System",
    description=(
        "CIS-based corpus lifecycle management: three-tier storage (Enhanced/Active/Archive), "
        "sleep-time curation, user-confirmed archival, and reinstatement tracking."
    ),
    config_schema=CorpusIntelligenceConfig,
    requires=[],
))
