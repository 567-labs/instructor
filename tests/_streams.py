"""Shared offline stream helpers for sync/async parametrized tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable
from typing import Any, TypeVar

T = TypeVar("T")


async def async_items(items: Iterable[T]) -> AsyncGenerator[T, None]:
    for item in items:
        yield item


async def consume(
    api: Any,
    chunks: Iterable[Any],
    is_async: bool,
    method: str = "model_from_chunks",
    **kwargs: Any,
) -> list[Any]:
    """Drain `api.<method>` (or `<method>_async` over an async source)."""
    if is_async:
        stream = getattr(api, method + "_async")(async_items(chunks), **kwargs)
        return [item async for item in stream]
    return list(getattr(api, method)(chunks, **kwargs))
