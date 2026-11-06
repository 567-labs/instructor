"""Compatibility facade for ``instructor.providers.cerebras.utils``."""

from instructor.providers._compat import make_getattr

__getattr__ = make_getattr("cerebras", ("client",))
