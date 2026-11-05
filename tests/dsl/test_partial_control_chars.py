"""Offline streaming contracts for context-aware control-character repair."""

import json
from collections.abc import AsyncGenerator, Iterable
from typing import Any, cast

import pytest
from jiter import from_json
from pydantic import BaseModel, Field, ValidationError, model_validator

from instructor.v2.dsl.partial import Partial


class Positive(BaseModel):
    n: int = Field(gt=0)


class PositiveValidator(BaseModel):
    n: int

    @model_validator(mode="after")
    def require_positive(self) -> "PositiveValidator":
        if self.n <= 0:
            raise ValueError("n must be positive")
        return self


class Nested(BaseModel):
    values: list[str]


class Payload(BaseModel):
    text: str
    nested: Nested
    labels: dict[str, str]


def chunkings(raw: str) -> Iterable[list[str]]:
    yield [raw]
    yield list(raw)
    for split in range(len(raw) + 1):
        yield [raw[:split], raw[split:]]


async def consume(model: type[BaseModel], chunks: list[Any], mode: str) -> list[Any]:
    api = cast(Any, Partial[model])
    if mode == "sync":
        return list(api.model_from_chunks(chunks))

    async def source() -> AsyncGenerator[Any, None]:
        for chunk in chunks:
            yield chunk

    return [value async for value in api.model_from_chunks_async(source())]


def outside_payloads(character: str, n: int) -> Iterable[str]:
    # Prefix, key, colon/value, token, comma, root closure, and suffix contexts.
    yield character + '{"n":' + str(n) + "}"
    yield "{" + character + '"n":' + str(n) + "}"
    yield '{"n"' + character + ":" + str(n) + "}"
    yield '{"n":' + character + str(n) + "}"
    yield '{"n":' + str(n) + character + "}"
    yield '{"n":' + str(n) + "," + character + '"extra":0}'
    yield '{"n":' + str(n) + "}" + character


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("code", [0x7F, 0x85, 0x9F])
async def test_outside_repair_preserves_completed_validation(
    code: int, mode: str
) -> None:
    character = chr(code)
    for raw in outside_payloads(character, 1):
        for chunks in chunkings(raw):
            models = await consume(Positive, chunks, mode)
            assert models[-1].model_dump() == {"n": 1}

    # Exhaust the complete stream, including its post-loop validation. Neither
    # before-close nor trailing DEL/C1 may turn n=-1 into a successful partial.
    for model in (Positive, PositiveValidator):
        for raw in ('{"n":-1' + character + "}", '{"n":-1}' + character):
            for chunks in chunkings(raw):
                with pytest.raises(ValidationError):
                    await consume(model, chunks, mode)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("code", [0x7F, 0x85, 0x9F])
async def test_nested_unicode_and_split_escapes(code: int, mode: str) -> None:
    character = chr(code)
    values = ["\\" * count + '"' + character + "雪🙂" for count in range(3)]
    # A value ending in an even run of encoded backslashes must close normally;
    # escaped quotes after odd runs must not end the surrounding JSON string.
    values += ["\\" * count + character for count in range(3)]
    expected = {
        "text": character + "\u2028\u2029é",
        "nested": {"values": values},
        "labels": {values[1]: values[2], character: character},
    }
    for ensure_ascii in (False, True):
        raw = json.dumps(expected, ensure_ascii=ensure_ascii, separators=(",", ":"))
        direct = Payload.model_validate_json(raw).model_dump()
        for chunks in chunkings(raw):
            models = await consume(Payload, chunks, mode)
            assert models[-1].model_dump() == direct


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_c0_removed_before_escape_tracking(mode: str) -> None:
    controls = "".join(chr(code) for code in range(0x20))
    expected = {
        "text": 'a\\"\x7fb',
        "nested": {"values": ['\\"\x85', "\\\\\x9f", "\x00\n\t"]},
        "labels": {'\\"\x7f': "\\\x85"},
    }
    clean = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
    # Even when raw C0 intervenes, the retained quote/backslash/escape sequence
    # must be exactly the same as clean JSON, regardless of transport splits.
    raw = clean.replace("\\", "\\" + controls)
    raw = controls + raw[:-1] + controls + "}" + controls + "\x7f"
    for chunks in chunkings(raw):
        models = await consume(Payload, chunks, mode)
        assert models[-1].model_dump() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("code", [0x00, 0x0A, 0x1F])
async def test_raw_and_escaped_c0_repair_unchanged(code: int, mode: str) -> None:
    class Text(BaseModel):
        text: str

    character = chr(code)
    for raw, expected in (
        ('{"text":"a' + character + 'b"}', "ab"),
        (json.dumps({"text": "a" + character + "b"}), "a" + character + "b"),
    ):
        for chunks in chunkings(raw):
            models = await consume(Text, chunks, mode)
            assert models[-1].text == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_genuinely_incomplete_streams_keep_partial_contract(mode: str) -> None:
    for raw in ('{"n":-1', '{"n":-1,', '{"n":', "{"):
        for chunks in chunkings(raw):
            models = await consume(Positive, chunks, mode)
            assert models[-1].n == (-1 if "-1" in raw else None)

    class Text(BaseModel):
        text: str
        n: int = Field(gt=0)

    for raw in (
        '{"n":-1,"text":"unfinished',
        '{"n":-1,"text":"backslash\\',
        '{"n":-1,"text":"escaped\\"',
    ):
        # Preserve the installed parser's behavior for an unfinished escape.
        expected = from_json(raw.encode(), partial_mode="trailing-strings").get("text")
        for chunks in chunkings(raw):
            models = await consume(Text, chunks, mode)
            assert models[-1].n == -1
            assert models[-1].text == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_stream_state_is_isolated_and_skipped_chunks_do_not_advance(
    mode: str,
) -> None:
    class Text(BaseModel):
        text: str

    class BadString:
        def __str__(self) -> str:
            raise ValueError("not stringifiable")

    api = cast(Any, Partial[Text])
    chunks_a = ['{"text":"\\', None, BadString(), '"\x7f', 0, '"}\x7f']
    chunks_b = ['\x7f{"text":', '"ok"}\x9f']
    if mode == "sync":
        first = api.model_from_chunks(chunks_a)
        second = api.model_from_chunks(chunks_b)
        next(first)
        next(second)
        models_b = list(second)
        models_a = list(first)
    else:

        async def source(chunks: list[Any]) -> AsyncGenerator[Any, None]:
            for chunk in chunks:
                yield chunk

        first = api.model_from_chunks_async(source(chunks_a))
        second = api.model_from_chunks_async(source(chunks_b))
        await first.__anext__()
        await second.__anext__()
        models_b = [value async for value in second]
        models_a = [value async for value in first]
    assert models_a[-1].text == '"\x7f0'
    assert models_b[-1].text == "ok"
    # A truncated stream must not leave its pending escape in the next stream.
    await consume(Text, ['{"text":"\\'], mode)
    assert (await consume(Text, ['\x7f{"text":"fresh"}\x9f'], mode))[-1].text == "fresh"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_invalid_escape_controls_do_not_return_unvalidated_models(
    mode: str,
) -> None:
    class TextPositive(BaseModel):
        text: str
        n: int = Field(gt=0)

    # A raw control after a single slash is an invalid escape, unlike a
    # valid encoded double slash followed by literal DEL/C1 string data.
    for control in ("\x7f", "\x85", "\x9f"):
        for suffix in ("n", '"', "\\"):
            raw = '{"text":"a\\' + control + suffix + 'z","n":-1}'
            for chunks in chunkings(raw):
                with pytest.raises(ValueError):
                    await consume(TextPositive, chunks, mode)
