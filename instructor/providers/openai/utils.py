"""Compatibility facade for ``instructor.providers.openai.utils``."""

from instructor.providers._compat import make_getattr

__getattr__ = make_getattr("openai", ("handlers", "schema", "multimodal", "templating"))
