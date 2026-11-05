from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from instructor.v2.core.json import (
    MAX_JSON_EXTRACTION_CHARS,
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
)
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.partial import Partial
from instructor.v2.providers.anthropic.handlers import AnthropicJSONHandler
from instructor.v2.providers.gemini.handlers import GeminiJSONHandler
from instructor.v2.providers.mistral.handlers import MistralMDJSONHandler
from instructor.v2.providers.openai.handlers import OpenAIMDJSONHandler
from instructor.v2.providers.xai.handlers import XAIMDJSONHandler
from tests.coverage._streams import async_items


class Answer(BaseModel):
    value: int


def event(content: str) -> SimpleNamespace:
    # Supply the documented text-delta shape for each provider.
    delta = SimpleNamespace(content=content, text=content)
    choices = [SimpleNamespace(delta=delta)]
    return SimpleNamespace(
        choices=choices,
        data=SimpleNamespace(choices=choices),
        delta=delta,
        text=content,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler",
    [
        OpenAIMDJSONHandler(),
        MistralMDJSONHandler(),
        XAIMDJSONHandler(),
        GeminiJSONHandler(),
        AnthropicJSONHandler(),
    ],
)
async def test_single_model_waits_for_the_last_valid_object(handler: Any) -> None:
    chunks = [
        'example {"wrong":true}\n```json\n{"value":1}\n```\n',
        'final {"value":',
        '2} {"unfinished":',
    ]
    assert extract_json_from_codeblock("".join(chunks)) == '{"value":2}'
    events = [event(chunk) for chunk in chunks]
    sync_results = list(
        handler.parse_response(iter(events), Partial[Answer], stream=True)
    )
    async_results = [
        item
        async for item in handler.parse_response(
            async_items(events), Partial[Answer], stream=True
        )
    ]
    assert [item.value for item in sync_results] == [2]
    assert [item.value for item in async_results] == [2]


@pytest.mark.asyncio
async def test_invalid_final_answer_triggers_validation_and_iterables_keep_all_items() -> (
    None
):
    handler = OpenAIMDJSONHandler()
    events = [event('{"value":1}{"wrong":true}')]
    with pytest.raises(ValidationError):
        handler.parse_response(iter(events), Partial[Answer], stream=True)
    with pytest.raises(ValidationError):
        async for _ in handler.parse_response(
            async_items(events), Partial[Answer], stream=True
        ):
            pass

    items = [event('[{"value":1}]'), event('[{"value":2}]')]
    iterable = IterableModel(Answer)
    assert list(handler.parse_response(iter(items), iterable, stream=True)) == [
        Answer(value=1),
        Answer(value=2),
    ]
    assert [
        item
        async for item in handler.parse_response(
            async_items(items), iterable, stream=True
        )
    ] == [Answer(value=1), Answer(value=2)]


@pytest.mark.asyncio
async def test_single_model_size_limit_and_incomplete_fallback() -> None:
    chunks = ["example ```json\n", '{"value":12']
    assert "".join(extract_json_from_stream(chunks, single_model=True)) == '{"value":12'
    assert (
        "".join(
            [
                part
                async for part in extract_json_from_stream_async(
                    async_items(chunks), single_model=True
                )
            ]
        )
        == '{"value":12'
    )
    oversized = [" " * MAX_JSON_EXTRACTION_CHARS, "x"]
    with pytest.raises(ValueError, match="1 MiB"):
        list(extract_json_from_stream(oversized, single_model=True))
    with pytest.raises(ValueError, match="1 MiB"):
        async for _ in extract_json_from_stream_async(
            async_items(oversized), single_model=True
        ):
            pass
