"""Iterable streaming releases real SDK responses on early exit and errors."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, cast

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError

import instructor
from instructor import Mode


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

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"path": self.path, "body": data})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if outcome == "cancel":
                prefix, suffix = body.split(b"data: [DONE]", 1)
                self.wfile.write(prefix)
                self.wfile.flush()
                release.wait()
                try:
                    self.wfile.write(b"data: [DONE]" + suffix)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.wfile.write(body)

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


def assert_request(calls: list[dict[str, Any]], mode: Mode) -> None:
    assert len(calls) == 1
    assert calls[0]["path"] == (
        "/v1/responses" if mode == Mode.RESPONSES_TOOLS else "/v1/chat/completions"
    )
    assert calls[0]["body"]["stream"] is True


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("outcome", ["close", "validation-error", "exhausted"])
def test_sync_iterable_closes_sdk_response(
    stream_endpoint: tuple[str, list[dict[str, Any]]], mode: Mode, outcome: str
) -> None:
    url, calls = stream_endpoint
    client = instructor.from_provider(
        "openai/local-contract", mode=mode, base_url=url, api_key="local-only"
    )
    assert client.client is not None
    raw: list[Any] = []
    # A completion observer may keep the SDK response alive after parsing ends.
    client.on("completion:response", raw.append)
    try:
        items = client.create_iterable(
            messages=[{"role": "user", "content": "extract"}],
            response_model=Item,
            max_retries=0,
        )
        if outcome == "close":
            assert next(items) == Item(value=1)
            assert not raw[0].response.is_closed
            items.close()
        elif outcome == "validation-error":
            with pytest.raises(ValidationError):
                next(items)
        else:
            assert list(items) == [Item(value=1), Item(value=2)]
        assert raw[0].response.is_closed
        assert_request(calls, mode)
    finally:
        for completion in raw:
            completion.close()
        client.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "outcome", ["close", "validation-error", "exhausted", "cancel"]
)
async def test_async_iterable_closes_sdk_response(
    stream_endpoint: tuple[str, list[dict[str, Any]]], mode: Mode, outcome: str
) -> None:
    url, calls = stream_endpoint
    client = instructor.from_provider(
        "openai/local-contract",
        mode=mode,
        base_url=url,
        api_key="local-only",
        async_client=True,
    )
    assert client.client is not None
    raw: list[Any] = []
    client.on("completion:response", raw.append)
    try:
        items = client.create_iterable(
            messages=[{"role": "user", "content": "extract"}],
            response_model=Item,
            max_retries=0,
        )
        if outcome == "close":
            assert await items.__anext__() == Item(value=1)
            assert not raw[0].response.is_closed
            await items.aclose()
        elif outcome == "validation-error":
            with pytest.raises(ValidationError):
                await items.__anext__()
        elif outcome == "cancel":
            assert await items.__anext__() == Item(value=1)
            assert await items.__anext__() == Item(value=2)
            started = asyncio.Event()

            async def read_next() -> Item:
                started.set()
                return await items.__anext__()

            pending = asyncio.create_task(read_next())
            await started.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            assert [item async for item in items] == [Item(value=1), Item(value=2)]
        assert raw[0].response.is_closed
        assert_request(calls, mode)
    finally:
        for completion in raw:
            await completion.close()
        await client.client.close()


@pytest.mark.asyncio
async def test_iterable_closes_async_generator_source() -> None:
    from instructor.v2.dsl.iterable import IterableModel

    finalized: list[bool] = []

    async def source() -> AsyncGenerator[str, None]:
        try:
            yield '{"tasks":[{"value":1}'
            yield ',{"value":2}]}'
        finally:
            finalized.append(True)

    async def extract(chunks: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        async for chunk in chunks:
            yield chunk

    model = cast(Any, IterableModel(Item))
    items = model.from_streaming_response_async(source(), stream_extractor=extract)
    assert await items.__anext__() == Item(value=1)
    await items.aclose()
    assert finalized == [True]


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
