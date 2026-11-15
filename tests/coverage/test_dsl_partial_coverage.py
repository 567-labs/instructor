from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from threading import Event, Lock
from typing import Any, Optional, Union, cast
import typing

import pytest
from pydantic import BaseModel

from instructor.v2.dsl.json_tracker import JsonCompleteness
from instructor.v2.dsl.partial import Partial, PartialBase
import instructor.v2.dsl.partial as partial_module
from tests._streams import consume


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


class Named(BaseModel):
    name: str


class DefaultChild(BaseModel):
    value: Optional[int] = None


class DefaultParent(BaseModel):
    name: str
    child: DefaultChild = DefaultChild(value=7)


class FixedTuple(BaseModel):
    items: tuple[Named, int]


class VariadicTuple(BaseModel):
    items: tuple[Named, ...]


def _last_partial(model: type[BaseModel], chunks: list[str]) -> Any:
    return list(cast(Any, Partial)[model].model_from_chunks(chunks))[-1]


def test_partial_missing_nested_model_uses_declared_default() -> None:
    assert _last_partial(DefaultParent, ['{"name":"a']).child == DefaultChild(value=7)


@pytest.mark.parametrize(
    ("model", "chunks", "expected"),
    [
        (FixedTuple, ['{"items":[{"name":"a"},', "3"], (Named(name="a"), 3)),
        (
            VariadicTuple,
            ['{"items":[{"name":"a"},{"name":"b'],
            (Named(name="a"), Named(name="b")),
        ),
    ],
    ids=["fixed", "variadic"],
)
def test_partial_tuple_snapshot_builds_models(
    model: type[BaseModel], chunks: list[str], expected: tuple[Any, ...]
) -> None:
    items = _last_partial(model, chunks).items

    assert isinstance(items, tuple)
    assert items == expected
    assert type(items[0]) is Named


sync_and_async = pytest.mark.parametrize(
    "is_async", [False, True], ids=["sync", "async"]
)
COMPLETE = (
    '{"optional_note":"complete","metadata":{},'
    '"featured":{"number":1,"state":"ready"},"items":[]}'
)


def strip_prefix(is_async: bool) -> Any:
    """Stream extractor removing a transport prefix, in the matching flavor."""
    if is_async:

        async def aextract(source: AsyncGenerator[str, None]) -> Any:
            async for chunk in source:
                yield chunk.removeprefix("payload:")

        return aextract

    def extract(source: Iterable[str]) -> Generator[str, None, None]:
        for chunk in source:
            yield chunk.removeprefix("payload:")

    return extract


@pytest.mark.asyncio
@sync_and_async
async def test_stream_accepts_string_like_chunks_and_skips_bad_chunks(
    is_async: bool,
) -> None:
    chunks = [
        None,
        UnprintableChunk(),
        '{"optional_note":"live","metadata":{"source":"stream"},',
        TextChunk(
            '"featured":{"number":1,"state":"ready"},'
            '"items":[{"number":2,"state":"pending"}]}'
        ),
    ]
    values = await consume(Partial[Envelope], chunks, is_async)

    assert len(values) == 2
    assert values[0].optional_note == "live"
    assert values[-1] == Envelope(
        optional_note="live",
        metadata={"source": "stream"},
        featured=Item(number=1, state=State.READY),
        items=[Item(number=2, state=State.PENDING)],
    )


@pytest.mark.asyncio
@sync_and_async
async def test_partial_stream_handles_empty_incomplete_and_unbound_models(
    monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    partial_envelope = Partial[Envelope]
    assert await consume(partial_envelope, [], is_async) == []

    incomplete = await consume(partial_envelope, ['{"optional_note":"open'], is_async)
    assert [value.optional_note for value in incomplete] == ["open"]

    monkeypatch.setattr(partial_envelope, "_original_model", None)
    complete = await consume(partial_envelope, [COMPLETE], is_async)
    assert [value.optional_note for value in complete] == ["complete"]


@pytest.mark.asyncio
@sync_and_async
async def test_streaming_response_requires_and_uses_an_extractor(
    is_async: bool,
) -> None:
    method = "from_streaming_response"
    with pytest.raises(ValueError, match="stream_extractor is required"):
        await consume(Partial[Envelope], [], is_async, method, stream_extractor=None)

    values = await consume(
        Partial[Envelope],
        ["payload:" + COMPLETE],
        is_async,
        method,
        stream_extractor=strip_prefix(is_async),
    )

    assert values == [
        Envelope(
            optional_note="complete",
            metadata={},
            featured=Item(number=1, state=State.READY),
            items=[],
        )
    ]


@pytest.mark.asyncio
@sync_and_async
async def test_json_extractor_delegates_and_rejects_a_missing_extractor(
    is_async: bool,
) -> None:
    chunks = ["payload:{", 'payload:"ok":true}']
    extracted = await consume(
        PartialBase,
        chunks,
        is_async,
        "extract_json",
        stream_extractor=strip_prefix(is_async),
    )

    assert extracted == ["{", '"ok":true}']
    with pytest.raises(ValueError, match="stream_extractor is required"):
        await consume(
            PartialBase, chunks, is_async, "extract_json", stream_extractor=None
        )


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
