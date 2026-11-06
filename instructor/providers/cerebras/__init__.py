"""Legacy Cerebras provider path backed by v2."""

from instructor.providers._compat import make_getattr

__getattr__ = make_getattr("cerebras", ("client",))
