from __future__ import annotations

import json
import sys
from collections.abc import AsyncGenerator, Iterable

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from instructor.v2.core.json import (
    MAX_JSON_DEPTH,
    MAX_JSON_EXTRACTION_CHARS,
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
)
from instructor.v2.providers.openai.handlers import OpenAIMDJSONHandler


async def _async_chunks(chunks: Iterable[str]) -> AsyncGenerator[str, None]:
    for chunk in chunks:
        yield chunk


async def _extract(chunks: list[str], is_async: bool) -> str:
    if is_async:
        raw = _async_chunks(chunks)
        extracted = extract_json_from_stream_async(raw)
        try:
            return "".join([char async for char in extracted])
        finally:
            await extracted.aclose()
            await raw.aclose()
    return "".join(extract_json_from_stream(chunks))


def _partition(text: str, chunk_size: int) -> list[str]:
    return [
        text[index : index + chunk_size] for index in range(0, len(text), chunk_size)
    ]


@pytest.mark.parametrize("root", ["array", "object"])
def test_codeblock_wraps_real_decoder_recursion_error(root: str) -> None:
    # CPython's C JSON decoder has a separate recursion limit on newer versions.
    depth = max(10_000, 2 * sys.getrecursionlimit())
    opener, closer = ("[", "]") if root == "array" else ('{"a":', "}")
    payload = opener * depth + "0" + closer * depth
    with pytest.raises(
        ValueError, match="JSON extraction nesting limit exceeded"
    ) as failure:
        extract_json_from_codeblock("```json\n" + payload + "\n```")
    assert isinstance(failure.value.__cause__, RecursionError)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("chunk_size", [MAX_JSON_EXTRACTION_CHARS + 1, 4096])
@pytest.mark.parametrize("framing", ["prose", "plain", "fenced"])
@pytest.mark.parametrize("extra_char", [False, True], ids=["at-limit", "over-limit"])
async def test_total_character_boundary(
    is_async: bool, chunk_size: int, framing: str, extra_char: bool
) -> None:
    if framing == "prose":
        prefix, suffix = "", ""
    elif framing == "plain":
        prefix, suffix = 'before {"text":"', '"} after'
    else:
        prefix, suffix = '```json\n{"text":"', '"}\n```'
    padding = "a" * (
        MAX_JSON_EXTRACTION_CHARS + int(extra_char) - len(prefix) - len(suffix)
    )
    text = prefix + padding + suffix
    chunks = _partition(text, chunk_size)

    if extra_char:
        with pytest.raises(ValueError, match="1 MiB character limit"):
            await _extract(chunks, is_async)
    else:
        expected = "" if framing == "prose" else '{"text":"' + padding + '"}'
        assert await _extract(chunks, is_async) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_limit_counts_characters_not_utf8_bytes(is_async: bool) -> None:
    payload = '{"text":"' + "\u00e9" * (MAX_JSON_EXTRACTION_CHARS - 11) + '"}'
    assert len(payload) == MAX_JSON_EXTRACTION_CHARS
    assert len(payload.encode("utf-8")) > MAX_JSON_EXTRACTION_CHARS
    assert await _extract(_partition(payload, 4096), is_async) == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_total_budget_does_not_reset_after_complete_object(
    is_async: bool,
) -> None:
    chunks = ['{"a":1}', " " * (MAX_JSON_EXTRACTION_CHARS - 7), '{"b":2}']
    with pytest.raises(ValueError, match="1 MiB character limit"):
        await _extract(chunks, is_async)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_size_guard_counts_tail_skipped_by_fence_recovery(is_async: bool) -> None:
    # A closing fence inside incomplete JSON breaks the current chunk scan.
    text = '```json\n{"a":1\n```' + " " * MAX_JSON_EXTRACTION_CHARS
    with pytest.raises(ValueError, match="1 MiB character limit"):
        await _extract([text], is_async)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("chunk_size", [1, 17, 4096])
@pytest.mark.parametrize("fenced", [False, True], ids=["plain", "fenced"])
@pytest.mark.parametrize("root", ["array", "object"])
async def test_depth_boundary_and_129th_opener_at_eof(
    is_async: bool, chunk_size: int, fenced: bool, root: str
) -> None:
    opener, closer = ("[", "]") if root == "array" else ('{"a":', "}")
    prefix = "```json\n" if fenced else ""
    complete = opener * MAX_JSON_DEPTH + "0" + closer * MAX_JSON_DEPTH
    suffix = "\n```" if fenced else ""
    assert await _extract(
        _partition(prefix + complete + suffix, chunk_size), is_async
    ) == (complete)

    incomplete = prefix + opener * MAX_JSON_DEPTH
    assert await _extract(_partition(incomplete, chunk_size), is_async) == (
        opener * MAX_JSON_DEPTH
    )

    # End at the offending opening delimiter, with no next character to trigger a check.
    too_deep = incomplete + opener[0]
    with pytest.raises(ValueError, match="128 level limit"):
        await _extract(_partition(too_deep, chunk_size), is_async)
    with pytest.raises(ValueError, match="128 level limit"):
        await _extract(
            _partition(
                prefix
                + opener * (MAX_JSON_DEPTH + 1)
                + "0"
                + closer * (MAX_JSON_DEPTH + 1),
                chunk_size,
            ),
            is_async,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("chunk_size", [1, 19, 4096])
async def test_quoted_delimiters_and_escaped_quotes_do_not_increase_depth(
    is_async: bool, chunk_size: int
) -> None:
    value = "{" * (MAX_JSON_DEPTH + 1) + '\\"```' + "]" * (MAX_JSON_DEPTH + 1)
    payload = (
        "[" * (MAX_JSON_DEPTH - 1)
        + json.dumps({"text": value})
        + "]" * (MAX_JSON_DEPTH - 1)
    )
    text = "```json\n" + payload + "\n```"
    assert await _extract(_partition(text, chunk_size), is_async) == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("chunk_size", [1, 13, 4096])
async def test_depth_resets_between_objects_without_dropping_candidates(
    is_async: bool, chunk_size: int
) -> None:
    first = "[" * MAX_JSON_DEPTH + "0" + "]" * MAX_JSON_DEPTH
    second = '{"text":"[quoted]"}'
    assert await _extract(_partition(first + second, chunk_size), is_async) == (
        first + second
    )


def test_sync_complete_object_emits_before_next_input_chunk() -> None:
    consumed: list[int] = []
    first, second = '{"a":1}', '{"b":2}'

    def chunks() -> Iterable[str]:
        consumed.append(1)
        yield first
        consumed.append(2)
        yield second

    extracted = extract_json_from_stream(chunks())
    assert "".join(next(extracted) for _ in first) == first
    assert consumed == [1]
    assert "".join(extracted) == second
    assert consumed == [1, 2]


@pytest.mark.asyncio
async def test_async_complete_object_emits_before_next_input_chunk() -> None:
    consumed: list[int] = []
    first, second = '{"a":1}', '{"b":2}'

    async def chunks() -> AsyncGenerator[str, None]:
        consumed.append(1)
        yield first
        consumed.append(2)
        yield second

    extracted = extract_json_from_stream_async(chunks())
    assert "".join([await extracted.__anext__() for _ in first]) == first
    assert consumed == [1]
    assert "".join([char async for char in extracted]) == second
    assert consumed == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "case", ["valid", "size-at-limit", "size", "depth-at-limit", "depth"]
)
async def test_bounds_propagate_through_raw_openai_sdk_chunks(
    is_async: bool, case: str
) -> None:
    first = '{"text":"[quoted]"}'
    second = '{"number":2}'
    text_chunks = ["```json\n", first, second, "\n```"]
    expected = first + second
    if case == "size-at-limit":
        text_chunks = [" " * (MAX_JSON_EXTRACTION_CHARS - len(first)), first]
        expected = first
    elif case == "size":
        text_chunks = [" " * MAX_JSON_EXTRACTION_CHARS, " "]
    elif case == "depth-at-limit":
        expected = "[" * MAX_JSON_DEPTH + "0" + "]" * MAX_JSON_DEPTH
        text_chunks = _partition(expected, 17)
    elif case == "depth":
        text_chunks = ["[" * MAX_JSON_DEPTH, "["]
    raw = [
        ChatCompletionChunk(
            id="local",
            created=0,
            model="local",
            object="chat.completion.chunk",
            choices=[
                Choice(index=0, delta=ChoiceDelta(content=text), finish_reason=None)
            ],
        )
        for text in text_chunks
    ]
    handler = OpenAIMDJSONHandler()

    async def async_raw() -> AsyncGenerator[ChatCompletionChunk, None]:
        for chunk in raw:
            yield chunk

    async def collect() -> str:
        if is_async:
            raw_stream = async_raw()
            extracted = handler.extract_streaming_json_async(raw_stream)
            try:
                return "".join([char async for char in extracted])
            finally:
                await extracted.aclose()
                await raw_stream.aclose()
        return "".join(handler.extract_streaming_json(raw))

    if case in {"valid", "size-at-limit", "depth-at-limit"}:
        assert await collect() == expected
    else:
        message = "1 MiB character limit" if case == "size" else "128 level limit"
        with pytest.raises(ValueError, match=message):
            await collect()
