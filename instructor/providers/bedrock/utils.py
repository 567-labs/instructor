"""Compatibility facade for ``instructor.providers.bedrock.utils``."""

from instructor.providers._compat import make_getattr

__getattr__ = make_getattr("bedrock", ("handlers",))
