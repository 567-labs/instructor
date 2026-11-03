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


def test_extract_json_from_stream_yields_all_objects_in_one_chunk() -> None:
    # A complete object must not swallow a second object sharing its chunk.
    chunks = ['{"a":1}{"b":2}']
    assert "".join(extract_json_from_stream(chunks)) == '{"a":1}{"b":2}'


def test_extract_json_from_stream_yields_all_objects_in_fenced_chunk() -> None:
    chunks = ['```json\n{"a":1}{"b":2}\n```']
    assert "".join(extract_json_from_stream(chunks)) == '{"a":1}{"b":2}'


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
async def test_extract_json_from_stream_async_yields_all_objects_in_chunk() -> None:
    async def chunks():
        yield '{"a":1}{"b":2}'

    assert "".join(
        [chunk async for chunk in extract_json_from_stream_async(chunks())]
    ) == ('{"a":1}{"b":2}')


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


def test_extract_json_from_codeblock_exact_1mb_boundary_allowed() -> None:
    from instructor.v2.core.json import MAX_JSON_EXTRACTION_CHARS
    pad_len = MAX_JSON_EXTRACTION_CHARS - 6
    content_1mb = '{"a":' + '0' * pad_len + '}'
    assert len(content_1mb) == MAX_JSON_EXTRACTION_CHARS
    assert extract_json_from_codeblock(content_1mb) == content_1mb


def test_extract_json_from_codeblock_leading_object_with_trailing_prose() -> None:
    # A JSON object starting at index 0 must be captured when trailing prose exists
    content = '{"only": 1} trailing prose'
    assert extract_json_from_codeblock(content) == '{"only": 1}'


def test_extract_json_from_codeblock_exact_work_budget_boundary() -> None:
    # At index 16, remaining_work exactly equals len(content) - index (120 == 120),
    # so remaining_work < len(content) - index is False, successfully decoding the valid JSON
    valid_json = '{"valid":1}'
    content = '{' * 16 + valid_json + ' ' * (136 - 16 - len(valid_json))
    assert len(content) == 136
    assert extract_json_from_codeblock(content) == valid_json


def test_extract_json_from_codeblock_multiple_valid_objects_span_offset_budget() -> None:
    # Multiple valid objects must only decrement by span length (end - index), not end + index
    content = " ".join(['{"x": 1}'] * 25)
    assert extract_json_from_codeblock(content) == '{"x": 1}'


def test_extract_json_from_codeblock_exact_128_depth_boundary_allowed() -> None:
    # MAX_JSON_DEPTH is 128. Exact depth of 128 dict nesting must succeed without error
    content = '{"a": ' * 127 + '{"a": 1}' + '}' * 127
    assert extract_json_from_codeblock(content) == content


def test_extract_json_from_codeblock_depth_exceeds_128_rejected() -> None:
    # Nesting depth > 128 must be rejected with ValueError
    content = '{"a": ' * 128 + '{"a": 1}' + '}' * 129
    with pytest.raises(ValueError, match="128 level limit"):
        extract_json_from_codeblock(content)
