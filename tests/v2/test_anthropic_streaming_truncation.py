"""Anthropic Iterable streams must report a token limit between complete items.

A stream can end with ``message_delta.stop_reason == "max_tokens"`` after every
item that reached the client was complete.  The non-streaming path raises
``IncompleteOutputException`` for that stop reason; the streaming path used to
keep the items and return normally, so callers could not tell the list was cut
short.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.providers.anthropic.handlers import AnthropicToolsHandler


class Item(BaseModel):
    name: str


def _chunks(stop_reason: Any) -> list[Any]:
    from anthropic.types import (
        InputJSONDelta,
        RawContentBlockDeltaEvent,
        RawMessageDeltaEvent,
    )
    from anthropic.types.raw_message_delta_event import Delta

    return [
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=InputJSONDelta(
                type="input_json_delta", partial_json='{"name": "Ada"}'
            ),
        ),
        RawMessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason=stop_reason, stop_sequence=None),
            usage={"output_tokens": 1},
        ),
    ]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    ("stop_reason", "raises"),
    [("max_tokens", True), ("end_turn", False)],
)
@pytest.mark.asyncio
async def test_iterable_reports_max_tokens_between_complete_items(
    is_async: bool, stop_reason: Any, raises: bool
) -> None:
    handler = AnthropicToolsHandler()
    model = IterableModel(Item)
    chunks = _chunks(stop_reason)
    received: list[Item] = []

    if is_async:

        async def events():
            for chunk in chunks:
                yield chunk

        if raises:
            with pytest.raises(IncompleteOutputException):
                async for item in handler.parse_response(events(), model, stream=True):
                    received.append(item)
        else:
            async for item in handler.parse_response(events(), model, stream=True):
                received.append(item)
    elif raises:
        with pytest.raises(IncompleteOutputException):
            for item in handler.parse_response(iter(chunks), model, stream=True):
                received.append(item)
    else:
        for item in handler.parse_response(iter(chunks), model, stream=True):
            received.append(item)

    assert received == [Item(name="Ada")]
