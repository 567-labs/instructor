"""Compatibility facade for ``instructor.providers.anthropic.utils``."""

from instructor.providers._compat import make_getattr

__getattr__ = make_getattr(
    "anthropic", ("handlers", "schema", "multimodal", "parallel", "templating", "usage")
)
