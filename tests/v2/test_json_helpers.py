from __future__ import annotations

import pytest

from instructor.v2.core.json import (
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
)


def test_extract_json_from_codeblock_returns_json_span() -> None:
    assert (
        extract_json_from_codeblock('prefix ```json\n{"name":"Ada"}\n``` suffix')
        == '{"name":"Ada"}'
    )


def test_extract_json_from_codeblock_preserves_array_root() -> None:
    assert (
        extract_json_from_codeblock('prefix ```json\n[{"name":"Ada"}]\n``` suffix')
        == '[{"name":"Ada"}]'
    )


def test_extract_json_from_codeblock_ignores_brackets_in_prose() -> None:
    assert (
        extract_json_from_codeblock(
            'See [section] for details.\n```json\n{"name":"Ada"}\n```\n'
        )
        == '{"name":"Ada"}'
    )


def test_extract_json_from_codeblock_handles_even_backslashes_before_quote() -> None:
    content = 'prefix {"path":"C:\\\\","name":"Ada"} suffix'

    assert extract_json_from_codeblock(content) == '{"path":"C:\\\\","name":"Ada"}'


def test_extract_json_from_codeblock_returns_last_top_level_candidate() -> None:
    content = (
        'The user supplied {"is_verified":true}. '
        'The result is {"person":{"name":"Ada"}}.'
    )

    assert extract_json_from_codeblock(content) == '{"person":{"name":"Ada"}}'


def test_extract_json_from_codeblock_preserves_nested_candidate() -> None:
    content = 'prefix {"person":{"name":"Ada"}} suffix'

    assert extract_json_from_codeblock(content) == '{"person":{"name":"Ada"}}'


def test_extract_json_from_stream_handles_plain_json() -> None:
    assert "".join(extract_json_from_stream(["before ", '{"a":', "1}", " after"])) == (
        '{"a":1}'
    )


def test_extract_json_from_stream_handles_fenced_json() -> None:
    assert (
        "".join(extract_json_from_stream(["```json\n", '{"a":"{ok}"}', "\n```"]))
        == '{"a":"{ok}"}'
    )


def test_extract_json_from_stream_handles_plain_array_root() -> None:
    assert (
        "".join(
            extract_json_from_stream(["before ", '[{"a":', "1},", '{"a":2}]', " after"])
        )
        == '[{"a":1},{"a":2}]'
    )


def test_extract_json_from_stream_handles_fenced_array_root() -> None:
    assert (
        "".join(extract_json_from_stream(["```json\n", '[{"a":"[ok]"}]', "\n```"]))
        == '[{"a":"[ok]"}]'
    )


def test_extract_json_from_stream_handles_even_backslashes_before_quote() -> None:
    chunks = ['prefix {"path":"C:', '\\\\","name":"Ada"} suffix']

    assert "".join(extract_json_from_stream(chunks)) == (
        '{"path":"C:\\\\","name":"Ada"}'
    )


def test_extract_json_from_stream_preserves_backticks_in_plain_string() -> None:
    chunks = ['{"md":"use ', "`", "pip install`", ' here"}']

    assert "".join(extract_json_from_stream(chunks)) == (
        '{"md":"use `pip install` here"}'
    )


def test_extract_json_from_stream_preserves_triple_backticks_in_plain_string() -> None:
    chunks = ['{"code":"', "```", "py```", '"}']

    assert "".join(extract_json_from_stream(chunks)) == '{"code":"```py```"}'


def test_extract_json_from_stream_returns_last_object() -> None:
    chunks = ['{"a":1}{"b":2}']
    assert "".join(extract_json_from_stream(chunks)) == '{"b":2}'


def test_extract_json_from_stream_returns_last_object_in_fenced() -> None:
    chunks = ['```json\n{"a":1}{"b":2}\n```']
    assert "".join(extract_json_from_stream(chunks)) == '{"b":2}'


def test_extract_json_from_stream_preserves_backticks_in_fenced_string() -> None:
    chunks = ["```json\n", '{"code":"`inline`"}', "\n```"]

    assert "".join(extract_json_from_stream(chunks)) == '{"code":"`inline`"}'


def test_extract_json_from_stream_discards_non_json_brace_span_before_payload() -> None:
    # MD_JSON prompts often produce a prose preamble before the actual payload,
    # and that preamble can itself contain a balanced pair of braces that is not
    # JSON (e.g. a parenthetical aside). A naive bracket-matching scan treats that
    # span as "the JSON" once it balances and emits it verbatim, so the real
    # payload that follows gets appended onto invalid JSON instead of replacing
    # it. The extractor should recognize the span isn't valid JSON, drop it, and
    # keep scanning for the real payload.
    chunks = [
        "I'll pull out the fields now. ",
        "{Note: keeping the original casing} ",
        "Here is the result: ",
        '{"name":"Ada","age":30}',
    ]

    assert "".join(extract_json_from_stream(chunks)) == '{"name":"Ada","age":30}'


@pytest.mark.asyncio
async def test_extract_json_from_stream_async_preserves_backticks_in_string() -> None:
    async def chunks():
        for chunk in ['{"md":"use ', "`", "pip install`", ' here"}']:
            yield chunk

    assert (
        "".join([chunk async for chunk in extract_json_from_stream_async(chunks())])
        == '{"md":"use `pip install` here"}'
    )


@pytest.mark.asyncio
async def test_extract_json_from_stream_async_handles_fenced_json() -> None:
    async def chunks():
        for chunk in ["```json\n", '{"a":1}', "\n```"]:
            yield chunk

    assert "".join(
        [chunk async for chunk in extract_json_from_stream_async(chunks())]
    ) == ('{"a":1}')


@pytest.mark.asyncio
async def test_extract_json_from_stream_async_handles_array_root() -> None:
    async def chunks():
        for chunk in ["prefix ", "[", '{"a":1}', "]", " suffix"]:
            yield chunk

    assert (
        "".join([chunk async for chunk in extract_json_from_stream_async(chunks())])
        == '[{"a":1}]'
    )


@pytest.mark.asyncio
async def test_extract_json_from_stream_async_returns_last_object() -> None:
    async def chunks():
        yield '{"a":1}{"b":2}'

    assert "".join(
        [chunk async for chunk in extract_json_from_stream_async(chunks())]
    ) == ('{"b":2}')


@pytest.mark.asyncio
async def test_extract_json_from_stream_async_handles_even_backslashes() -> None:
    async def chunks():
        for chunk in ['prefix {"path":"C:', '\\\\","name":"Ada"} suffix']:
            yield chunk

    assert (
        "".join([chunk async for chunk in extract_json_from_stream_async(chunks())])
        == '{"path":"C:\\\\","name":"Ada"}'
    )


@pytest.mark.asyncio
async def test_extract_json_from_stream_async_discards_non_json_brace_span_before_payload() -> (
    None
):
    async def chunks():
        for chunk in [
            "I'll pull out the fields now. ",
            "{Note: keeping the original casing} ",
            "Here is the result: ",
            '{"name":"Ada","age":30}',
        ]:
            yield chunk

    assert (
        "".join([chunk async for chunk in extract_json_from_stream_async(chunks())])
        == '{"name":"Ada","age":30}'
    )


def test_stream_extraction_matches_codeblock_last_object() -> None:
    text = 'prefix {"a": 1} suffix {"a": 2}'
    assert extract_json_from_codeblock(text) == '{"a": 2}'
    assert "".join(extract_json_from_stream(list(text))) == '{"a": 2}'


@pytest.mark.asyncio
async def test_stream_async_extraction_matches_codeblock_last_object() -> None:
    text = 'prefix {"a": 1} suffix {"a": 2}'
    assert extract_json_from_codeblock(text) == '{"a": 2}'

    async def chunks():
        for c in list(text):
            yield c

    assert "".join([chunk async for chunk in extract_json_from_stream_async(chunks())]) == '{"a": 2}'
