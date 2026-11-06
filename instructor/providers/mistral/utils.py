"""Compatibility facade for ``instructor.providers.mistral.utils``."""

from instructor.providers._compat import make_getattr

__getattr__ = make_getattr("mistral", ("handlers", "multimodal"))
