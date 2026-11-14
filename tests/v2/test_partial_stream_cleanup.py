"""Async Partial streaming closes its source when the consumer stops early."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import json
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel
import pytest

import instructor
from instructor import Mode
from instructor.v2.dsl.partial import Partial
from tests import openai_http as httpx


class Item(BaseModel):
    name: str = ""
    value: int = 0


@pytest.mark.asyncio
async def test_partial_closes_async_generator_source() -> None:
    finalized: list[bool] = []

    async def source() -> AsyncGenerator[str, None]:
        try:
            yield '{"name": "a"'
            yield ', "value": 1}'
        finally:
            finalized.append(True)

    async def extract(chunks: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        async for chunk in chunks:
            yield chunk

    model: Any = Partial[Item]
    items = model.from_streaming_response_async(source(), stream_extractor=extract)
    first = await items.__anext__()
    assert first.name == "a"
    await items.aclose()

    assert finalized == [True]


async def _sse_body() -> AsyncGenerator[bytes, None]:
    # A lazy body keeps the response open until the SDK stream is closed.
    for fragment in ['{"name": "a"', ', "value": 1}']:
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
    requests: list[Any] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body(),
        )

    sdk = AsyncOpenAI(
        api_key="local-only",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
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
    first = await items.__anext__()
    assert first.name == "a"
    assert not raw[0].response.is_closed
    await items.aclose()

    assert raw[0].response.is_closed
    assert len(requests) == 1
    await sdk.close()
