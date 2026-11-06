"""Compatibility facade for ``instructor.providers.gemini.utils``."""

from instructor.providers._compat import make_getattr

__getattr__ = make_getattr("gemini", ("utils", "handlers", "schema", "templating"))
