"""Resolve the optional Mistral SDK without importing provider handlers."""

from __future__ import annotations

import importlib
from typing import Any


def load_mistral_client_type() -> type[Any] | None:
    for name in ("mistralai.client", "mistralai"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        client = getattr(module, "Mistral", None)
        if isinstance(client, type):
            return client
    return None
