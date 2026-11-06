"""Compatibility exports for v2-owned multimodal helpers."""

from instructor.v2.core import multimodal as _multimodal
from instructor.v2.core.multimodal import *  # noqa: F403

# Preserve the historical patching surface for callers that still target the
# compatibility module directly.
requests = _multimodal.requests
