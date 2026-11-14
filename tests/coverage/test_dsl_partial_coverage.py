from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from threading import Event, Lock
from typing import Any, Optional, Protocol, Union, cast
import typing

import pytest
from pydantic import BaseModel

from instructor.v2.dsl.json_tracker import JsonCompleteness
from instructor.v2.dsl.partial import Partial, PartialBase
import instructor.v2.dsl.partial as partial_module


class State(str, Enum):
    READY = "ready"
    PENDING = "pending"


class Item(BaseModel):
    number: int
    state: State


class Envelope(BaseModel):
    optional_note: Optional[str]
    metadata: dict[str, Union[str, int]]
    featured: Item
    items: list[Item]


class LooseEnvelope(BaseModel):
    metadata: dict
    values: typing.List  # noqa: UP006
    counts: list[int]


def test_partial_parser_accepts_non_trailing_string_mode() -> None:
    model = cast(Any, Partial[Item]).get_partial_model()
    parse = cast(Any, partial_module).process_potential_object
    item = parse('{"number": 7', True, model)
    assert item.number == 7
    assert item.state is None


class EnvelopeStream(Protocol):
    @classmethod
    def model_from_chunks(
        cls, json_chunks: Iterable[Any], **kwargs: Any
    ) -> Generator[Envelope, None, None]: ...

    @classmethod
    def model_from_chunks_async(
        cls, json_chunks: AsyncGenerator[Any, None], **kwargs: Any
    ) -> AsyncGenerator[Envelope, None]: ...

    @classmethod
    def from_streaming_response(
        cls,
        completion: Iterable[Any],
        stream_extractor: Optional[
            Callable[[Iterable[Any]], Generator[str, None, None]]
        ],
    ) -> Generator[Envelope, None, None]: ...

    @classmethod
    def from_streaming_response_async(
        cls,
        completion: AsyncGenerator[Any, None],
        stream_extractor: Optional[
            Callable[[AsyncGenerator[Any, None]], AsyncGenerator[str, None]]
        ],
    ) -> AsyncGenerator[Envelope, None]: ...


_build_partial_object = cast(
    Callable[[Any, type[BaseModel], JsonCompleteness, str], Any],
    vars(partial_module)["_build_partial_object"],
)
_unwrap_optional_base_model = cast(
    Callable[[Any], Optional[type[BaseModel]]],
    vars(partial_module)["_unwrap_optional_base_model"],
)


class TextChunk:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class UnprintableChunk:
    def __str__(self) -> str:
        raise ValueError("cannot render chunk")


def test_optional_base_model_unwrap_rejects_optional_scalars() -> None:
    assert _unwrap_optional_base_model(Optional[int]) is None


def test_partial_builder_validates_complete_nested_values_and_keeps_open_values() -> (
    None
):
    tracker = JsonCompleteness()
    tracker.analyze(
        '{"optional_note":null,"metadata":{"source":"stream","attempt":2},'
        '"featured":{"number":"7","state":"ready"},'
        '"items":[{"number":"8","state":"pending"},'
        '{"number":"9","state":"rea'
    )

    result = _build_partial_object(
        {
            "optional_note": None,
            "metadata": {"source": "stream", "attempt": 2},
            "featured": {"number": "7", "state": "ready"},
            "items": [
                {"number": "8", "state": "pending"},
                {"number": "9", "state": "rea"},
            ],
        },
        Envelope,
        tracker,
        "",
    )

    assert result.optional_note is None
    assert result.metadata == {"source": "stream", "attempt": 2}
    assert result.featured == Item(number=7, state=State.READY)
    assert result.items[0] == Item(number=8, state=State.PENDING)
    # Open (incomplete) list items are recursed into partial model instances,
    # consistent with how a singular open nested field is handled (see
    # test_partial_builder_recurses_into_open_nested_model_and_handles_scalars):
    # unvalidated/partial values are kept, but the item stays a typed instance
    # instead of degrading to a raw dict.
    assert isinstance(result.items[1], Item)
    assert result.items[1].number == "9"
    assert result.items[1].state == "rea"


def test_partial_builder_recurses_into_open_nested_model_and_handles_scalars() -> None:
    tracker = JsonCompleteness()
    tracker.analyze('{"featured":{"number":"12","state":"pen')

    result = _build_partial_object(
        {"featured": {"number": "12", "state": "pen"}}, Envelope, tracker, ""
    )

    assert result.featured.number == "12"
    assert result.featured.state == "pen"
    assert result.optional_note is None
    assert result.metadata is None
    assert result.items is None
    assert _build_partial_object(None, Envelope, tracker, "") is None
    assert _build_partial_object("still streaming", Envelope, tracker, "") == (
        "still streaming"
    )


def test_partial_builder_preserves_bare_containers_and_complete_scalar_items() -> None:
    tracker = JsonCompleteness()
    tracker.analyze('{"metadata":{"source":"stream"},"values":["raw"],"counts":[1]}')

    result = _build_partial_object(
        {"metadata": {"source": "stream"}, "values": ["raw"], "counts": [1]},
        LooseEnvelope,
        tracker,
        "",
    )

    assert result.metadata == {"source": "stream"}
    assert result.values == ["raw"]
    assert result.counts == [1]


def test_sync_stream_accepts_string_like_chunks_and_skips_bad_chunks() -> None:
    partial_envelope = cast(type[EnvelopeStream], Partial[Envelope])
    values = list(
        partial_envelope.model_from_chunks(
            [
                None,
                UnprintableChunk(),
                '{"optional_note":"live","metadata":{"source":"sync"},',
                TextChunk(
                    '"featured":{"number":1,"state":"ready"},'
                    '"items":[{"number":2,"state":"pending"}]}'
                ),
            ]
        )
    )

    assert len(values) == 2
    assert values[0].optional_note == "live"
    assert values[-1] == Envelope(
        optional_note="live",
        metadata={"source": "sync"},
        featured=Item(number=1, state=State.READY),
        items=[Item(number=2, state=State.PENDING)],
    )


def test_sync_partial_stream_handles_empty_incomplete_and_unbound_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial_envelope = cast(type[EnvelopeStream], Partial[Envelope])
    assert list(partial_envelope.model_from_chunks([])) == []

    incomplete = list(partial_envelope.model_from_chunks(['{"optional_note":"open']))
    assert len(incomplete) == 1
    assert incomplete[0].optional_note == "open"

    monkeypatch.setattr(partial_envelope, "_original_model", None)
    complete = list(
        partial_envelope.model_from_chunks(
            [
                '{"optional_note":"complete","metadata":{},'
                '"featured":{"number":1,"state":"ready"},"items":[]}'
            ]
        )
    )
    assert len(complete) == 1
    assert complete[0].optional_note == "complete"


@pytest.mark.asyncio
async def test_async_stream_accepts_string_like_chunks_and_skips_bad_chunks() -> None:
    async def chunks() -> AsyncGenerator[Any, None]:
        yield None
        yield UnprintableChunk()
        yield '{"optional_note":"live","metadata":{"source":"async"},'
        yield TextChunk(
            '"featured":{"number":4,"state":"pending"},'
            '"items":[{"number":5,"state":"ready"}]}'
        )

    partial_envelope = cast(type[EnvelopeStream], Partial[Envelope])
    values = [
        value async for value in partial_envelope.model_from_chunks_async(chunks())
    ]

    assert len(values) == 2
    assert values[0].metadata == {"source": "async"}
    assert values[-1] == Envelope(
        optional_note="live",
        metadata={"source": "async"},
        featured=Item(number=4, state=State.PENDING),
        items=[Item(number=5, state=State.READY)],
    )


@pytest.mark.asyncio
async def test_async_partial_stream_handles_empty_incomplete_and_unbound_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def chunks(*values: str) -> AsyncGenerator[str, None]:
        for value in values:
            yield value

    partial_envelope = cast(type[EnvelopeStream], Partial[Envelope])
    assert [
        value async for value in partial_envelope.model_from_chunks_async(chunks())
    ] == []

    incomplete = [
        value
        async for value in partial_envelope.model_from_chunks_async(
            chunks('{"optional_note":"open')
        )
    ]
    assert len(incomplete) == 1
    assert incomplete[0].optional_note == "open"

    monkeypatch.setattr(partial_envelope, "_original_model", None)
    complete = [
        value
        async for value in partial_envelope.model_from_chunks_async(
            chunks(
                '{"optional_note":"complete","metadata":{},'
                '"featured":{"number":1,"state":"ready"},"items":[]}'
            )
        )
    ]
    assert len(complete) == 1
    assert complete[0].optional_note == "complete"


def test_sync_streaming_response_requires_and_uses_an_extractor() -> None:
    partial_envelope = cast(type[EnvelopeStream], Partial[Envelope])
    with pytest.raises(ValueError, match="stream_extractor is required"):
        list(partial_envelope.from_streaming_response([], None))

    def extractor(completion: Iterable[str]) -> Generator[str, None, None]:
        yield from completion

    values = list(
        partial_envelope.from_streaming_response(
            [
                '{"optional_note":"extracted","metadata":{},'
                '"featured":{"number":1,"state":"ready"},"items":[]}'
            ],
            extractor,
        )
    )

    assert values == [
        Envelope(
            optional_note="extracted",
            metadata={},
            featured=Item(number=1, state=State.READY),
            items=[],
        )
    ]


@pytest.mark.asyncio
async def test_async_streaming_response_requires_and_uses_an_extractor() -> None:
    partial_envelope = cast(type[EnvelopeStream], Partial[Envelope])

    async def empty() -> AsyncGenerator[str, None]:
        if False:
            yield ""

    with pytest.raises(ValueError, match="stream_extractor is required"):
        [
            value
            async for value in partial_envelope.from_streaming_response_async(
                empty(), None
            )
        ]

    async def completion() -> AsyncGenerator[str, None]:
        yield (
            '{"optional_note":"extracted","metadata":{},'
            '"featured":{"number":6,"state":"pending"},"items":[]}'
        )

    async def extractor(source: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        async for chunk in source:
            yield chunk

    values = [
        value
        async for value in partial_envelope.from_streaming_response_async(
            completion(), extractor
        )
    ]

    assert values == [
        Envelope(
            optional_note="extracted",
            metadata={},
            featured=Item(number=6, state=State.PENDING),
            items=[],
        )
    ]


def test_sync_json_extractor_delegates_and_rejects_a_missing_extractor() -> None:
    def extractor(completion: Iterable[str]) -> Generator[str, None, None]:
        for chunk in completion:
            yield chunk.removeprefix("payload:")

    assert list(
        PartialBase.extract_json(["payload:{", 'payload:"ok":true}'], extractor)
    ) == ["{", '"ok":true}']
    with pytest.raises(ValueError, match="stream_extractor is required"):
        list(PartialBase.extract_json([], None))


@pytest.mark.asyncio
async def test_async_json_extractor_delegates_and_rejects_a_missing_extractor() -> None:
    async def completion() -> AsyncGenerator[str, None]:
        yield "payload:{"
        yield 'payload:"ok":true}'

    async def extractor(source: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        async for chunk in source:
            yield chunk.removeprefix("payload:")

    assert [
        chunk async for chunk in PartialBase.extract_json_async(completion(), extractor)
    ] == ["{", '"ok":true}']
    with pytest.raises(ValueError, match="stream_extractor is required"):
        [chunk async for chunk in PartialBase.extract_json_async(completion(), None)]


@pytest.mark.asyncio
async def test_async_responses_extractor_ignores_unhandled_events() -> None:
    from openai.types.responses import (
        ResponseFunctionCallArgumentsDeltaEvent,
        ResponseReasoningSummaryTextDeltaEvent,
    )

    from instructor.v2.core.mode import Mode

    async def events() -> AsyncGenerator[Any, None]:
        yield ResponseReasoningSummaryTextDeltaEvent(
            delta="thinking",
            item_id="reasoning-1",
            output_index=0,
            sequence_number=1,
            summary_index=0,
            type="response.reasoning_summary_text.delta",
        )
        yield object()
        yield ResponseFunctionCallArgumentsDeltaEvent(
            delta='{"ok":true}',
            item_id="call-1",
            output_index=1,
            sequence_number=2,
            type="response.function_call_arguments.delta",
        )

    assert [
        chunk
        async for chunk in PartialBase.extract_json_async(
            events(), Mode.RESPONSES_TOOLS
        )
    ] == ['{"ok":true}']


def test_partial_rejects_direct_use_and_wraps_a_recursive_model() -> None:
    with pytest.raises(TypeError, match="Cannot instantiate abstract Partial class"):
        cast(Callable[[], object], Partial)()

    with pytest.raises(TypeError, match="Cannot subclass .*Partial"):
        type("InvalidPartial", (cast(type, Partial),), {})

    class RecursiveNode(BaseModel):
        child: "RecursiveNode"

    RecursiveNode.model_rebuild()
    schema = Partial[RecursiveNode].model_json_schema()

    assert schema["$defs"]["PartialRecursiveNode"]["properties"]["child"] == {
        "$ref": "#/$defs/RecursiveNode"
    }
    assert schema["$defs"]["RecursiveNode"]["properties"]["child"] == {
        "$ref": "#/$defs/RecursiveNode"
    }


def test_partial_processing_isolated_between_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Address(BaseModel):
        street: str

    class Home(BaseModel):
        addresses: list[Address]

    class Office(BaseModel):
        addresses: list[Address]

    first_entered = Event()
    second_entered = Event()
    release_first = Event()
    call_lock = Lock()
    calls = 0
    partial_module_any = cast(Any, partial_module)
    original_make_partial_type = cast(
        Callable[..., type[BaseModel]],
        partial_module_any._make_partial_type,
    )

    def block_first_address_conversion(
        annotation: type[BaseModel], *, make_fields_optional: bool = False
    ) -> type[BaseModel]:
        nonlocal calls
        if annotation is Address:
            with call_lock:
                calls += 1
                call_number = calls

            if call_number == 1:
                first_entered.set()
                assert release_first.wait(timeout=1)
            elif call_number == 2:
                second_entered.set()

        return original_make_partial_type(
            annotation, make_fields_optional=make_fields_optional
        )

    monkeypatch.setattr(
        partial_module, "_make_partial_type", block_first_address_conversion
    )
    make_partial_type = cast(
        Callable[..., type[BaseModel]],
        partial_module_any._make_partial_type,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(make_partial_type, Home, make_fields_optional=True)
        assert first_entered.wait(timeout=1)

        second = executor.submit(make_partial_type, Office, make_fields_optional=True)
        assert second_entered.wait(timeout=1)
        release_first.set()

        home_partial = first.result()
        office_partial = second.result()

    assert "PartialAddress" in home_partial.model_json_schema()["$defs"]
    assert "PartialAddress" in office_partial.model_json_schema()["$defs"]
