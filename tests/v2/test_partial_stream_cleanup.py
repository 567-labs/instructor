"""Partial streaming releases connections when a consumer stops parsing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator, Iterable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import product
import json
import threading
from typing import Any, cast

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError

import instructor
from instructor import Mode
from instructor.core import InstructorRetryException


class Item(BaseModel):
    value: int = Field(ge=0)


MODES = [
    Mode.TOOLS,
    Mode.JSON,
    Mode.JSON_SCHEMA,
    Mode.RESPONSES_TOOLS,
]


def stream_body(mode: Mode, invalid: bool) -> bytes:
    value = -1 if invalid else 1
    fragments = [f'{{"value":{value}', "}"]
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
                                "name": "PartialItem",
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
    return "".join(events).encode()


@pytest.fixture
def stream_endpoint(
    mode: Mode, outcome: str
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    calls: list[dict[str, Any]] = []
    release = threading.Event()
    prefix = stream_body(mode, outcome == "validation-error")
    suffix = b"data: [DONE]\n\n"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            calls.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(prefix) + len(suffix)))
            self.end_headers()
            self.wfile.write(prefix)
            self.wfile.flush()
            if outcome != "exhausted" and mode != Mode.MD_JSON:
                release.wait(10)
            try:
                self.wfile.write(suffix)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ARG002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("outcome", ["close"])
async def test_closed_partial_releases_single_connection_pool(
    stream_endpoint: tuple[str, list[dict[str, Any]]],
    mode: Mode,
) -> None:
    url, calls = stream_endpoint
    transport = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1),
        timeout=httpx.Timeout(5, pool=0.1),
        trust_env=False,
    )
    client = instructor.from_provider(
        "openai/local-contract",
        mode=mode,
        base_url=url,
        api_key="local-only",
        http_client=transport,
        max_retries=0,
        async_client=True,
    )
    assert client.client is not None
    raw: list[Any] = []
    client.on("completion:response", raw.append)
    try:
        for _ in range(2):
            snapshots = client.create_partial(
                messages=[{"role": "user", "content": "extract"}],
                response_model=Item,
                max_retries=0,
            )
            assert await anext(snapshots) == Item(value=1)
            await snapshots.aclose()
        assert len(calls) == 2
        assert all(call["stream"] for call in calls)
        assert all(completion.response.is_closed for completion in raw)
    finally:
        for completion in raw:
            await completion.close()
        await client.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "outcome"),
    list(product(MODES, ["close", "validation-error", "exhausted", "cancel"]))
    + [(Mode.MD_JSON, "exhausted"), (Mode.MD_JSON, "validation-error")],
)
async def test_async_partial_closes_sdk_response(
    stream_endpoint: tuple[str, list[dict[str, Any]]], mode: Mode, outcome: str
) -> None:
    url, calls = stream_endpoint
    client = instructor.from_provider(
        "openai/local-contract",
        mode=mode,
        base_url=url,
        api_key="local-only",
        async_client=True,
        max_retries=0,
        timeout=5,
    )
    assert client.client is not None
    raw: list[Any] = []
    client.on("completion:response", raw.append)
    snapshots = client.create_partial(
        messages=[{"role": "user", "content": "extract"}],
        response_model=Item,
        max_retries=0,
    )
    try:
        if outcome == "validation-error":
            with pytest.raises(ValidationError):
                async for _ in snapshots:
                    pass
        elif outcome == "exhausted":
            results = [snapshot async for snapshot in snapshots]
            assert results[-1] == Item(value=1)
        else:
            assert await anext(snapshots) == Item(value=1)
            assert not raw[0].response.is_closed
            if outcome == "close":
                await snapshots.aclose()
            else:
                assert await anext(snapshots) == Item(value=1)
                started = asyncio.Event()

                async def read_next() -> Item:
                    started.set()
                    return await anext(snapshots)

                pending = asyncio.create_task(read_next())
                await started.wait()
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
        assert raw[0].response.is_closed
        assert len(calls) == 1
        assert calls[0]["stream"] is True
    finally:
        await snapshots.aclose()
        for completion in raw:
            await completion.close()
        await client.client.close()


@pytest.mark.parametrize("mode", [*MODES, Mode.MD_JSON])
@pytest.mark.parametrize("outcome", ["validation-error", "exhausted"])
def test_sync_partial_closes_sdk_response(
    stream_endpoint: tuple[str, list[dict[str, Any]]], mode: Mode, outcome: str
) -> None:
    url, calls = stream_endpoint
    client = instructor.from_provider(
        "openai/local-contract",
        mode=mode,
        base_url=url,
        api_key="local-only",
        max_retries=0,
        timeout=5,
    )
    assert client.client is not None
    raw: list[Any] = []
    client.on("completion:response", raw.append)
    try:
        if outcome == "validation-error":
            with pytest.raises(InstructorRetryException) as caught:
                client.create_partial(
                    messages=[{"role": "user", "content": "extract"}],
                    response_model=Item,
                    max_retries=0,
                )
            assert isinstance(caught.value.__cause__, ValidationError)
        else:
            snapshots = client.create_partial(
                messages=[{"role": "user", "content": "extract"}],
                response_model=Item,
                max_retries=0,
            )
            assert list(snapshots)[-1] == Item(value=1)
        assert raw[0].response.is_closed
        assert len(calls) == 1
        assert calls[0]["stream"] is True
    finally:
        for completion in raw:
            completion.close()
        client.client.close()


def test_partial_closes_generator_source() -> None:
    from instructor import Partial

    finalized: list[bool] = []

    def source() -> Generator[str, None, None]:
        try:
            yield '{"value":1'
            yield "}"
        finally:
            finalized.append(True)

    def extract(chunks: Iterable[str]) -> Generator[str, None, None]:
        yield from chunks

    model = cast(Any, Partial[Item])
    snapshots = model.from_streaming_response(source(), stream_extractor=extract)
    assert next(snapshots) == Item(value=1)
    snapshots.close()
    assert finalized == [True]


@pytest.mark.asyncio
async def test_partial_closes_async_generator_source() -> None:
    from instructor import Partial

    finalized: list[bool] = []

    async def source() -> AsyncGenerator[str, None]:
        try:
            yield '{"value":1'
            yield "}"
        finally:
            finalized.append(True)

    async def extract(chunks: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        async for chunk in chunks:
            yield chunk

    model = cast(Any, Partial[Item])
    snapshots = model.from_streaming_response_async(source(), stream_extractor=extract)
    assert await anext(snapshots) == Item(value=1)
    await snapshots.aclose()
    assert finalized == [True]
