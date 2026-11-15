"""Iterable and Partial streaming release real SDK responses on early exit and errors."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator
import json
import threading
from typing import Any, cast

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError

import instructor
from instructor import Mode
from tests._http import JSONHandler, serve


class Item(BaseModel):
    value: int = Field(ge=0)


MODES = [
    Mode.TOOLS,
    Mode.JSON,
    Mode.JSON_SCHEMA,
    Mode.MD_JSON,
    Mode.RESPONSES_TOOLS,
]


def stream_body(mode: Mode, invalid: bool) -> bytes:
    first = -1 if invalid else 1
    fragments = [f'{{"tasks":[{{"value":{first}}}', ',{"value":2}]}']
    if mode == Mode.MD_JSON:
        fragments = ["```json\n" + fragments[0], fragments[1] + "\n```"]
    events = []
    for index, fragment in enumerate(fragments):
        if mode == Mode.RESPONSES_TOOLS:
            event = {
                "type": "response.function_call_arguments.delta",
                "item_id": "item_local",
                "output_index": 0,
                "sequence_number": index,
                "delta": fragment,
            }
        else:
            delta = (
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_local",
                            "type": "function",
                            "function": {
                                "name": "IterableItem",
                                "arguments": fragment,
                            },
                        }
                    ]
                }
                if mode == Mode.TOOLS
                else {"content": fragment}
            )
            event = {
                "id": "chatcmpl-local",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "local-contract",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        events.append("data: " + json.dumps(event) + "\n\n")
    return ("".join(events) + "data: [DONE]\n\n").encode()


@pytest.fixture
def stream_endpoint(
    mode: Mode, outcome: str
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    calls: list[dict[str, Any]] = []
    body = stream_body(mode, outcome == "validation-error")
    release = threading.Event()

    class Handler(JSONHandler):
        def do_POST(self) -> None:
            calls.append({"path": self.path, "body": self.read_json()})
            if outcome != "cancel":
                self.send_body(body, content_type="text/event-stream")
                return
            # Send all events except [DONE], then hold the stream open.
            prefix, suffix = body.split(b"data: [DONE]", 1)
            self.send_body(prefix, content_type="text/event-stream")
            self.wfile.flush()
            release.wait()

    with serve(Handler) as url:
        try:
            yield f"{url}/v1", calls
        finally:
            release.set()


def assert_request(calls: list[dict[str, Any]], mode: Mode) -> None:
    assert len(calls) == 1
    assert calls[0]["path"] == (
        "/v1/responses" if mode == Mode.RESPONSES_TOOLS else "/v1/chat/completions"
    )
    assert calls[0]["body"]["stream"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "is_async,outcome",
    [
        pytest.param(
            is_async, outcome, id=f"{outcome}-{'async' if is_async else 'sync'}"
        )
        for is_async in (False, True)
        for outcome in ("close", "validation-error", "exhausted", "cancel")
        if is_async or outcome != "cancel"
    ],
)
async def test_iterable_closes_sdk_response(
    stream_endpoint: tuple[str, list[dict[str, Any]]],
    mode: Mode,
    is_async: bool,
    outcome: str,
) -> None:
    url, calls = stream_endpoint
    client: Any = instructor.from_provider(
        "openai/local-contract",
        mode=mode,
        base_url=url,
        api_key="local-only",
        async_client=is_async,
    )
    assert client.client is not None
    raw: list[Any] = []
    # A completion observer may keep the SDK response alive after parsing ends.
    client.on("completion:response", raw.append)
    items = client.create_iterable(
        messages=[{"role": "user", "content": "extract"}],
        response_model=Item,
        max_retries=0,
    )

    async def step() -> Item:
        return await items.__anext__() if is_async else next(items)

    try:
        if outcome == "close":
            assert await step() == Item(value=1)
            assert not raw[0].response.is_closed
            await items.aclose() if is_async else items.close()
        elif outcome == "validation-error":
            with pytest.raises(ValidationError):
                await step()
        elif outcome == "cancel":
            assert [await step(), await step()] == [Item(value=1), Item(value=2)]
            started = asyncio.Event()

            async def read_next() -> Item:
                started.set()
                return await step()

            pending = asyncio.create_task(read_next())
            await started.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            items_list = [i async for i in items] if is_async else list(items)
            assert items_list == [Item(value=1), Item(value=2)]
        assert raw[0].response.is_closed
        assert_request(calls, mode)
    finally:
        for completion in raw:
            await completion.close() if is_async else completion.close()
        await client.client.close() if is_async else client.client.close()


class _Source:
    """Async chunk source without any close hook."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> _Source:
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class _SyncClosingSource(_Source):
    """Async chunk source whose only close hook is synchronous."""

    closed = False

    def close(self) -> None:
        self.closed = True


async def _passthrough(chunks: Any) -> AsyncGenerator[str, None]:
    async for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("source_kind", ["async-generator", "sync-close", "no-close"])
@pytest.mark.parametrize(
    ("dsl", "chunks"),
    [
        ("partial", ['{"value":1', "}"]),
        ("iterable", ['{"tasks":[{"value":1}', ',{"value":2}]}']),
    ],
)
async def test_async_dsl_stream_closes_source_on_early_exit(
    dsl: str, chunks: list[str], source_kind: str
) -> None:
    from instructor.v2.dsl.iterable import IterableModel
    from instructor.v2.dsl.partial import Partial

    finalized: list[bool] = []

    async def generator() -> AsyncGenerator[str, None]:
        try:
            for chunk in chunks:
                yield chunk
        finally:
            finalized.append(True)

    source: Any = {
        "async-generator": generator,
        "sync-close": lambda: _SyncClosingSource(chunks),
        "no-close": lambda: _Source(chunks),
    }[source_kind]()
    model = cast(Any, Partial[Item] if dsl == "partial" else IterableModel(Item))
    items = model.from_streaming_response_async(source, stream_extractor=_passthrough)

    assert (await items.__anext__()).value == 1
    await items.aclose()

    if source_kind == "async-generator":
        assert finalized == [True]
    elif source_kind == "sync-close":
        assert source.closed


async def _partial_sse_body() -> AsyncGenerator[bytes, None]:
    # A lazy body keeps the response open until the SDK stream is closed.
    for fragment in ['{"value": 1', "}"]:
        event = {
            "id": "chatcmpl-local",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "local",
            "choices": [
                {"index": 0, "delta": {"content": fragment}, "finish_reason": None}
            ],
        }
        yield ("data: " + json.dumps(event) + "\n\n").encode()
    yield b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_async_create_partial_closes_sdk_stream_on_early_exit() -> None:
    from openai import AsyncOpenAI

    from tests import openai_http

    requests: list[Any] = []

    def transport(request: openai_http.Request) -> openai_http.Response:
        requests.append(request)
        return openai_http.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_partial_sse_body(),
        )

    sdk = AsyncOpenAI(
        api_key="local-only",
        http_client=openai_http.AsyncClient(
            transport=openai_http.MockTransport(transport)
        ),
    )
    client = instructor.from_openai(sdk, mode=Mode.JSON)
    raw: list[Any] = []
    client.on("completion:response", raw.append)

    items = client.create_partial(
        response_model=Item,
        model="local",
        messages=[{"role": "user", "content": "extract"}],
        max_retries=0,
    )
    assert (await items.__anext__()).value == 1
    assert not raw[0].response.is_closed
    await items.aclose()

    assert raw[0].response.is_closed
    assert len(requests) == 1
    await sdk.close()


@pytest.mark.parametrize("mode", [Mode.TOOLS])
@pytest.mark.parametrize("outcome", ["reuse"])
def test_closed_iterable_releases_single_connection_pool(
    stream_endpoint: tuple[str, list[dict[str, Any]]],
) -> None:
    url, calls = stream_endpoint
    transport = httpx.Client(
        limits=httpx.Limits(max_connections=1),
        timeout=httpx.Timeout(5, pool=0.1),
        trust_env=False,
    )
    client = instructor.from_provider(
        "openai/local-contract",
        base_url=url,
        api_key="local-only",
        http_client=transport,
        max_retries=0,
    )
    assert client.client is not None
    raw: list[Any] = []
    client.on("completion:response", raw.append)
    try:
        for _ in range(2):
            items = client.create_iterable(
                messages=[{"role": "user", "content": "extract"}],
                response_model=Item,
                max_retries=0,
            )
            assert next(items) == Item(value=1)
            items.close()
        assert len(calls) == 2
        assert all(completion.response.is_closed for completion in raw)
    finally:
        for completion in raw:
            completion.close()
        client.client.close()
