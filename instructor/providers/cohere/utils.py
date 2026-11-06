"""Compatibility facade for ``instructor.providers.cohere.utils``."""

from instructor.providers._compat import make_getattr

__getattr__ = make_getattr("cohere", ("handlers", "templating"))
