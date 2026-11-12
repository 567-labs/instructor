"""Cohere terminal events must not silently complete an Iterable."""

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from pydantic import BaseModel

cohere = pytest.importorskip("cohere")

from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.partial import Partial
from instructor.v2.providers.cohere.handlers import CohereToolsHandler


class Item(BaseModel):
    count: int


def events(text: str, reason: str | None) -> list[Any]:
    return [
        cohere.ChatContentDeltaEvent(
            type="content-delta",
            index=0,
            delta={"message": {"content": {"text": text}}},
        ),
        cohere.ChatMessageEndEvent(type="message-end", delta=None),
        cohere.ChatMessageEndEvent(type="message-end", delta={"usage": {}}),
        cohere.ChatMessageEndEvent(type="message-end", delta={"finish_reason": reason}),
    ]


@pytest.mark.parametrize(
    "text", ['{"tasks":[{"count":3}', '{"tasks":[{"count":3}]}', '{"tasks":[']
)
@pytest.mark.parametrize("reason", ["MAX_TOKENS", "COMPLETE", "STOP_SEQUENCE", None])
def test_iterable_finish_reason(text: str, reason: str | None) -> None:
    chunks = events(text, reason)
    stream = CohereToolsHandler().parse_response(
        iter(chunks), IterableModel(Item), stream=True
    )
    if "count" in text:
        assert next(stream).count == 3
    if reason == "MAX_TOKENS":
        with pytest.raises(IncompleteOutputException) as error:
            list(stream)
        assert error.value.last_completion is chunks[-1]
    else:
        assert list(stream) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["MAX_TOKENS", "COMPLETE", None])
@pytest.mark.parametrize("text", ['{"tasks":[{"count":3}', '{"tasks":[{"count":3}]}'])
async def test_async_iterable_finish_reason(text: str, reason: str | None) -> None:
    chunks = events(text, reason)

    async def source() -> AsyncGenerator[Any, None]:
        for chunk in chunks:
            yield chunk

    stream = CohereToolsHandler().parse_response(
        source(), IterableModel(Item), stream=True, is_async=True
    )
    assert (await anext(stream)).count == 3
    if reason == "MAX_TOKENS":
        with pytest.raises(IncompleteOutputException) as error:
            [item async for item in stream]
        assert error.value.last_completion is chunks[-1]
    else:
        assert [item async for item in stream] == []


def test_partial_retains_incomplete_generation() -> None:
    result = list(
        CohereToolsHandler().parse_response(
            iter(events('{"count":3}', "MAX_TOKENS")), Partial[Item], stream=True
        )
    )
    assert result[-1].count == 3
