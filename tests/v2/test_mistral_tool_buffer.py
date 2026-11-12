"""Empty extractor events and unfinished task wrappers need explicit handling."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.dsl.iterable import IterableBase, IterableModel
from instructor.v2.providers.mistral._streaming import (
    tasks_from_mistral_chunks,
    tasks_from_mistral_chunks_async,
)


class Item(BaseModel):
    name: str


@pytest.mark.parametrize("chunks", [[], [None, "", {}]])
def test_empty_extractor_events_keep_their_previous_behavior(chunks: Any) -> None:
    assert (
        list(
            tasks_from_mistral_chunks(
                cast(type[IterableBase], IterableModel(Item)), iter(chunks)
            )
        )
        == []
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[], [None, "", {}]])
async def test_async_empty_extractor_events_keep_their_previous_behavior(
    chunks: Any,
) -> None:
    async def stream() -> AsyncIterator[Any]:
        for chunk in chunks:
            yield chunk

    assert [
        item
        async for item in tasks_from_mistral_chunks_async(
            cast(type[IterableBase], IterableModel(Item)), stream()
        )
    ] == []


@pytest.mark.parametrize("text", ['{"tasks":[{"name":"Ada"}]', '{"tasks":["'])
def test_unfinished_wrapper_cannot_silently_complete(text: str) -> None:
    with pytest.raises(IncompleteOutputException):
        list(
            tasks_from_mistral_chunks(
                cast(type[IterableBase], IterableModel(Item)), iter(text)
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ['{"tasks":[{"name":"Ada"}]', '{"tasks":["'])
async def test_async_unfinished_wrapper_cannot_silently_complete(text: str) -> None:
    async def stream() -> AsyncIterator[str]:
        for char in text:
            yield char

    with pytest.raises(IncompleteOutputException):
        [
            item
            async for item in tasks_from_mistral_chunks_async(
                cast(type[IterableBase], IterableModel(Item)), stream()
            )
        ]
