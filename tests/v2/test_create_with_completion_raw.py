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


def _transport(paths: list[str], payload: dict[str, Any]) -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        assert "tools" not in json.loads(request.content)
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handle)


@pytest.mark.parametrize(("mode", "path", "payload", "completion_type"), CASES)
def test_sync_create_with_completion_without_model(
    mode: Mode, path: str, payload: dict[str, Any], completion_type: type
) -> None:
    paths: list[str] = []
    sdk = OpenAI(
        api_key="local-only",
        http_client=httpx.Client(transport=_transport(paths, payload)),
    )
    client = instructor.from_openai(sdk, mode=mode)

    result, completion = client.create_with_completion(
        messages=[{"role": "user", "content": "hi"}],
        response_model=None,
        model="local",
    )

    assert isinstance(completion, completion_type)
    assert result is completion
    assert paths == [path]
    sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "path", "payload", "completion_type"), CASES)
async def test_async_create_with_completion_without_model(
    mode: Mode, path: str, payload: dict[str, Any], completion_type: type
) -> None:
    paths: list[str] = []
    sdk = AsyncOpenAI(
        api_key="local-only",
        http_client=httpx.AsyncClient(transport=_transport(paths, payload)),
    )
    client = instructor.from_openai(sdk, mode=mode)

    result, completion = await client.create_with_completion(
        messages=[{"role": "user", "content": "hi"}],
        response_model=None,
        model="local",
    )

    assert isinstance(completion, completion_type)
    assert result is completion
    assert paths == [path]
    await sdk.close()
