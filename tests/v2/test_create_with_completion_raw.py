"""create_with_completion(response_model=None) returns the raw completion twice."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from openai.types.responses import Response
import pytest

import instructor
from instructor import Mode
from tests import openai_http as httpx

CHAT_PAYLOAD: dict[str, Any] = {
    "id": "chatcmpl-local",
    "object": "chat.completion",
    "created": 1,
    "model": "local",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
}

RESPONSES_PAYLOAD: dict[str, Any] = {
    "id": "resp_local",
    "object": "response",
    "created_at": 1,
    "model": "local",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg_local",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}],
        }
    ],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}

CASES = [
    (Mode.TOOLS, "/v1/chat/completions", CHAT_PAYLOAD, ChatCompletion),
    (Mode.RESPONSES_TOOLS, "/v1/responses", RESPONSES_PAYLOAD, Response),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(("mode", "path", "payload", "completion_type"), CASES)
async def test_create_with_completion_without_model_returns_raw_completion(
    mode: Mode,
    path: str,
    payload: dict[str, Any],
    completion_type: type,
    use_async: bool,
) -> None:
    paths: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        assert "tools" not in json.loads(request.content)
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handle)
    sdk: Any = (
        AsyncOpenAI(
            api_key="local-only", http_client=httpx.AsyncClient(transport=transport)
        )
        if use_async
        else OpenAI(api_key="local-only", http_client=httpx.Client(transport=transport))
    )
    client = instructor.from_openai(sdk, mode=mode)

    call = client.create_with_completion(
        messages=[{"role": "user", "content": "hi"}],
        response_model=None,
        model="local",
    )
    result, completion = await call if use_async else call

    assert isinstance(completion, completion_type)
    assert result is completion
    assert paths == [path]
    closed = sdk.close()
    if use_async:
        await closed
