"""Disable ChromaDB telemetry that causes a capture() signature bug and high CPU."""
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")

try:
    import chromadb.telemetry.product.posthog as _posthog
    _posthog.Posthog.capture = lambda *args, **kwargs: None
    _posthog.Posthog._direct_capture = lambda *args, **kwargs: None
except Exception:
    pass

try:
    import chromadb.config as _config
    _orig_init = _config.Settings.__init__
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.anonymized_telemetry = False
    _config.Settings.__init__ = _patched_init
except Exception:
    pass

# Posthog v3+ changed capture signature; ChromaDB's old 3-positional call
# (user_id, event_name, properties) breaks. Patch to accept the old signature.
try:
    import posthog as _ph
    _orig_capture = _ph.capture
    def _patched_capture(a, b=None, c=None, **kwargs):
        # Old ChromaDB call: posthog.capture(user_id, event_name, properties)
        if b is not None and c is not None:
            return _orig_capture(event=b, distinct_id=a, properties=c, **kwargs)
        if b is not None:
            return _orig_capture(event=b, distinct_id=a, **kwargs)
        return _orig_capture(event=a, **kwargs)
    _ph.capture = _patched_capture
except Exception:
    pass
