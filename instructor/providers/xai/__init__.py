"""Legacy xAI provider path backed by v2."""

from instructor.providers._compat import make_getattr

__getattr__ = make_getattr("xai", ("client", "handlers"))
