from __future__ import annotations

from . import handlers  # noqa: F401 - Import to trigger handler registration

try:
    from .client import from_genai
except ImportError:
    from_genai = None

__all__ = ["from_genai"]
