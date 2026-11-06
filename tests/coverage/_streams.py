from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable
from typing import TypeVar

T = TypeVar("T")


async def async_items(items: Iterable[T]) -> AsyncGenerator[T, None]:
    for item in items:
        yield item
