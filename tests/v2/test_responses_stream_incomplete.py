"""Typed Responses events must not turn an incomplete extraction into success."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, cast

from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseIncompleteEvent,
    ResponseTextDeltaEvent,
)
from pydantic import BaseModel
import pytest

from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.partial import Partial
from instructor.v2.providers.openai.handlers import OpenAIResponsesToolsHandler


class Item(BaseModel):
    value: int


def response(status: str, reason: str | None) -> Response:
    return Response.model_validate(
        {
            "id": "resp_local",
            "created_at": 1,
            "model": "local-contract",
            "object": "response",
            "status": status,
            "incomplete_details": {"reason": reason} if reason else None,
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def delta(text: str, channel: str) -> Any:
    fields = {
        "delta": text,
        "item_id": "item_local",
        "output_index": 0,
        "sequence_number": 0,
    }
    if channel == "text":
        return ResponseTextDeltaEvent(
            type="response.output_text.delta", content_index=0, logprobs=[], **fields
        )
    return ResponseFunctionCallArgumentsDeltaEvent(
        type="response.function_call_arguments.delta", **fields
    )


async def async_events(events: list[Any]) -> AsyncGenerator[Any, None]:
    for event in events:
        yield event


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("channel", ["function", "text"])
@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter", None])
@pytest.mark.parametrize(
    "payload", ["", '{"tasks":[{"value":1}', '{"tasks":[{"value":1}]}']
)
async def test_incomplete_iterable_keeps_items_and_reports_terminal_response(
    is_async: bool, channel: str, reason: str | None, payload: str
) -> None:
    terminal = response("incomplete", reason)
    events = [delta(payload, channel)] if payload else []
    events.append(
        ResponseIncompleteEvent(
            type="response.incomplete", sequence_number=1, response=terminal
        )
    )
    handler = OpenAIResponsesToolsHandler()
    model = cast(Any, IterableModel(Item))
    received = []
    with pytest.raises(IncompleteOutputException) as caught:
        if is_async:
            async for item in handler.parse_response(
                async_events(events), model, stream=True
            ):
                received.append(item)
        else:
            for item in handler.parse_response(iter(events), model, stream=True):
                received.append(item)
    assert received == ([Item(value=1)] if payload else [])
    assert caught.value.last_completion is terminal
    assert reason is None or reason in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("partial", "status"),
    [(False, "completed"), (True, "completed"), (True, "incomplete")],
)
async def test_completed_iterables_and_partial_contract(
    is_async: bool, partial: bool, status: str
) -> None:
    terminal = response(status, "max_output_tokens" if status == "incomplete" else None)
    terminal_event = (
        ResponseIncompleteEvent(
            type="response.incomplete", sequence_number=1, response=terminal
        )
        if status == "incomplete"
        else ResponseCompletedEvent(
            type="response.completed", sequence_number=1, response=terminal
        )
    )
    payload = '{"value":1}' if partial else '{"tasks":[{"value":1}]}'
    model = Partial[Item] if partial else IterableModel(Item)
    events = [
        delta(payload, "function"),
        terminal_event,
    ]
    handler = OpenAIResponsesToolsHandler()
    if is_async:
        items = [
            item
            async for item in handler.parse_response(
                async_events(events), model, stream=True
            )
        ]
    else:
        items = list(handler.parse_response(iter(events), model, stream=True))
    assert items[-1].value == 1
