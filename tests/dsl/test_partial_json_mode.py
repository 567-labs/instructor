from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from datetime import date, datetime
from decimal import Decimal
from typing import Any, TypeVar, cast
from uuid import UUID

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    create_model,
    field_validator,
)

from instructor import Partial
from instructor.v2.core.json import extract_json_from_stream

T_Model = TypeVar("T_Model", bound=BaseModel)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"]),
]


async def parse(
    schema: type[T_Model], chunks: list[str], asynchronous: bool, **kwargs: Any
) -> list[T_Model]:
    api = cast(Any, Partial[schema])
    if asynchronous:

        async def source() -> AsyncGenerator[str, None]:
            for chunk in chunks:
                yield chunk

        return [obj async for obj in api.model_from_chunks_async(source(), **kwargs)]
    return list(api.model_from_chunks(chunks, **kwargs))


@pytest.mark.parametrize(
    ("typ", "value"),
    [
        (date, "2026-09-30"),
        (datetime, "2026-09-30T09:00:00Z"),
        (UUID, "12345678-1234-5678-1234-567812345678"),
        (tuple[int, int], [1, 2]),
    ],
)
@pytest.mark.parametrize("strict_source", ["argument", "config", "field"])
async def test_complete_strict_json_matches_direct(
    typ: Any, value: Any, strict_source: str, asynchronous: bool
) -> None:
    schema = create_model(
        "Value",
        __config__=ConfigDict(strict=strict_source == "config"),
        value=(typ, Field(strict=True) if strict_source == "field" else ...),
    )
    kwargs: dict[str, Any] = {"strict": True} if strict_source == "argument" else {}
    payload = json.dumps({"value": value})
    expected = schema.model_validate_json(payload, **kwargs)
    result = await parse(schema, [payload[:-1], "}"], asynchronous, **kwargs)
    assert result[-1] == expected


class Event(BaseModel):
    day: date


class Envelope(BaseModel):
    event: Event
    note: str


class ListEnvelope(BaseModel):
    events: list[Event]
    note: str


@pytest.mark.parametrize(
    ("schema", "payload", "attribute"),
    [
        (Envelope, '{"event":{"day":"2026-09-30"},"note":"d', "event"),
        (ListEnvelope, '{"events":[{"day":"2026-09-30"}],"note":"d', "events"),
    ],
)
async def test_complete_nested_json_keeps_json_strictness(
    schema: type[BaseModel], payload: str, attribute: str, asynchronous: bool
) -> None:
    result = await parse(schema, [payload], asynchronous, strict=True)
    nested = getattr(result[-1], attribute)
    if isinstance(nested, list):
        nested = nested[0]
    assert isinstance(nested, Event)
    assert nested.day == date(2026, 9, 30)
    assert cast(Envelope | ListEnvelope, result[-1]).note == "d"


async def test_validation_info_remains_json_mode(asynchronous: bool) -> None:
    class Checked(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def check_mode(cls, value: int, info: ValidationInfo) -> int:
            assert info.mode == "json"
            assert info.context == {"marker": "kept"}
            return value

    payload = '{"value":1}'
    expected = Checked.model_validate_json(payload, context={"marker": "kept"})
    result = await parse(Checked, [payload], asynchronous, context={"marker": "kept"})
    assert result[-1] == expected


async def test_strict_rejects_numeric_string_control(asynchronous: bool) -> None:
    class Number(BaseModel):
        value: int

    payload = '{"value":"1"}'
    with pytest.raises(ValidationError):
        Number.model_validate_json(payload, strict=True)
    with pytest.raises(ValidationError):
        await parse(Number, [payload], asynchronous, strict=True)


async def test_lax_json_control(asynchronous: bool) -> None:
    payload = '{"day":"2026-09-30"}'
    result = await parse(Event, [payload], asynchronous, strict=False)
    assert result[-1] == Event.model_validate_json(payload, strict=False)


async def test_incomplete_fields_remain_unvalidated(asynchronous: bool) -> None:
    result = await parse(Event, ['{"day":"2026-'], asynchronous, strict=True)
    assert result[-1].day == "2026-"


async def test_missing_required_fields_still_fail(asynchronous: bool) -> None:
    with pytest.raises(ValidationError, match="day"):
        await parse(Event, ["{}"], asynchronous, strict=True)


async def test_trailing_whitespace_keeps_json_mode(asynchronous: bool) -> None:
    payload = '{"day":"2026-09-30"}'
    result = await parse(Event, [payload, "  \n\t"], asynchronous, strict=True)
    assert result[-1] == Event.model_validate_json(payload, strict=True)


async def test_nested_number_precision_matches_json_validation(
    asynchronous: bool,
) -> None:
    class Numbers(BaseModel):
        value: Decimal
        count: int

    class NumbersEnvelope(BaseModel):
        numbers: Numbers
        note: str

    payload = (
        '{"numbers":{"value":1.2345678901234567890123456789,'
        '"count":1234567890123456789012345678901234567890},"note":"done"}'
    )
    expected = NumbersEnvelope.model_validate_json(payload, strict=True)
    result = await parse(
        NumbersEnvelope, [payload[:-4], payload[-4:]], asynchronous, strict=True
    )
    assert result[0].numbers == expected.numbers
    assert result[-1] == expected


@pytest.mark.parametrize("is_list", [False, True])
async def test_nested_validation_context_and_mode(
    is_list: bool, asynchronous: bool
) -> None:
    class Checked(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def check_mode(cls, value: int, info: ValidationInfo) -> int:
            assert info.mode == "json"
            assert info.context == {"marker": "kept"}
            return value

    schema = create_model(
        "CheckedEnvelope",
        item=(list[Checked] if is_list else Checked, ...),
        note=(str, ...),
    )
    item = [{"value": 1}] if is_list else {"value": 1}
    payload = json.dumps({"item": item, "note": "done"})
    result = await parse(
        schema,
        [payload[:-4], payload[-4:]],
        asynchronous,
        strict=True,
        context={"marker": "kept"},
    )
    first_item = cast(Any, result[0]).item
    nested = first_item[0] if is_list else first_item
    assert nested.value == 1
    assert result[-1] == schema.model_validate_json(
        payload, strict=True, context={"marker": "kept"}
    )


async def test_extracted_fenced_json_keeps_strictness(asynchronous: bool) -> None:
    payload = '{"day":"2026-09-30"}'
    chunks = list(extract_json_from_stream(["```json\n", payload, "\n``` "]))
    result = await parse(Event, chunks, asynchronous, strict=True)
    assert result[-1] == Event.model_validate_json(payload, strict=True)
