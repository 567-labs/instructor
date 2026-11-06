"""Compatibility facade for ``instructor.providers.perplexity.utils``."""

from instructor.providers._compat import make_getattr

__getattr__ = make_getattr("perplexity", ("handlers",))
