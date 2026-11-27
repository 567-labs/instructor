from __future__ import annotations

from functools import partial
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
@pytest.mark.parametrize("fenced", [False, True])
async def test_md_json_grouping_preserves_public_chunks_and_final_selection(
    fenced: bool,
) -> None:
    handler = OpenAIMDJSONHandler()
    candidates = ['{"value":1}', '{"tasks":[{"value":2}]}']
    payload = "".join(candidates)
    text = f'{{"ignore":1}} ```json\n{payload}\n```' if fenced else payload
    events = [event(char) for char in text]

    for group, single, expected in [
        (False, False, list(payload)),
        (True, False, candidates),
        (True, True, [candidates[-1]]),
    ]:
        stream = handler.extract_streaming_json(
            events, single_model=single, _group_candidates=group
        )
        assert list(stream) == expected
        async_stream = handler.extract_streaming_json_async(
            async_items(events), single_model=single, _group_candidates=group
        )
        assert [part async for part in async_stream] == expected


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
        list(handler.parse_response(iter(events), Partial[Answer], stream=True))
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefix", ['user said {"value":99}', '{"value":99} before the answer']
)
async def test_iterable_prefers_fenced_items_over_prose(prefix):
    from instructor.v2.core.json import (
        extract_json_from_stream,
        extract_json_from_stream_async,
    )

    fence = "```"
    text = (
        prefix
        + "\n"
        + fence
        + 'json\n[{"value":1},{"value":2}]\n'
        + fence
        + ' after {"value":88}'
    )
    expected_text = '[{"value":1},{"value":2}]'
    assert "".join(extract_json_from_stream(iter(text))) == expected_text
    assert (
        "".join(
            [
                part
                async for part in extract_json_from_stream_async(
                    async_items(list(text))
                )
            ]
        )
        == expected_text
    )

    handler = OpenAIMDJSONHandler()
    model = IterableModel(Answer)
    events = [event(part) for part in text]
    expected = [Answer(value=1), Answer(value=2)]
    assert list(handler.parse_response(iter(events), model, stream=True)) == expected
    assert [
        item
        async for item in handler.parse_response(
            async_items(events), model, stream=True
        )
    ] == expected


def test_fenced_items_are_available_before_consuming_later_chunks():
    fence = "```"
    seen = []

    def chunks():
        seen.append(1)
        yield fence + 'json\n{"value":1}'
        seen.append(2)
        yield '{"value":2}\n' + fence

    result = extract_json_from_stream(chunks())
    assert "".join(next(result) for _ in '{"value":1}') == '{"value":1}'
    assert seen == [1]
    assert "".join(result) == '{"value":2}'


class NestedAnswer(BaseModel):
    value: int
    child: Answer


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["", "```json\n"])
async def test_partial_stream_keeps_completed_child_inside_incomplete_root(
    prefix: str,
) -> None:
    chunks = [prefix + '{"value":1,"child":', '{"value":2}']
    model: Any = Partial[NestedAnswer]
    sync_results = list(
        model.from_streaming_response(
            iter(chunks),
            stream_extractor=partial(extract_json_from_stream, single_model=True),
        )
    )
    async_results = [
        item
        async for item in model.from_streaming_response_async(
            async_items(chunks),
            stream_extractor=partial(extract_json_from_stream_async, single_model=True),
        )
    ]

    expected = {"value": 1, "child": {"value": 2}}
    assert sync_results[-1].model_dump() == expected
    assert async_results[-1].model_dump() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"items":[{"value":2}]', '{"items":[{"value":2}]'),
        ('{"value":1} then {"child":{"value":2}', '{"value":1}'),
        ('prose [broken {"value":2}', '{"value":2}'),
        ('{"value":1} then {"value":2}', '{"value":2}'),
    ],
)
async def test_single_model_selection_preserves_container_boundaries(
    text: str, expected: str
) -> None:
    assert "".join(extract_json_from_stream(text, single_model=True)) == expected
    assert (
        "".join(
            [
                part
                async for part in extract_json_from_stream_async(
                    async_items(list(text)), single_model=True
                )
            ]
        )
        == expected
    )
