"""Compatibility facade for ``instructor.providers.writer.utils``."""

from instructor.providers._compat import make_getattr

__getattr__ = make_getattr("writer", ("handlers",))
