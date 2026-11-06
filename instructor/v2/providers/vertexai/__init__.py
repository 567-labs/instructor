"""VertexAI v2 provider handlers and client."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "VertexAIJSONHandler",
    "VertexAIParallelToolsHandler",
    "VertexAIToolsHandler",
    "from_vertexai",
]

_LAZY_ATTRS = {
    "from_vertexai": (".client", "from_vertexai"),
    "VertexAIJSONHandler": (".handlers", "VertexAIJSONHandler"),
    "VertexAIParallelToolsHandler": (
        ".handlers",
        "VertexAIParallelToolsHandler",
    ),
    "VertexAIToolsHandler": (".handlers", "VertexAIToolsHandler"),
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    module_path, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_path, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
