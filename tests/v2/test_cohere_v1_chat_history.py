"""Check the Cohere V1 chat history sent by the public client."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

import instructor
from instructor import Mode

cohere = pytest.importorskip("cohere")


class Answer(BaseModel):
    answer: str


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON])
def test_v1_chat_history_roles_on_initial_request_and_reask(mode: Mode) -> None:
    requests: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        text = "{}" if len(requests) == 1 else '{"answer": "London"}'
        return httpx.Response(200, json={"text": text, "finish_reason": "COMPLETE"})

    messages = [
        {"role": "system", "content": "Answer with one city."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"},
        {"role": "user", "content": "What about the UK?"},
    ]
    expected_history = [
        {"role": "SYSTEM", "message": "Answer with one city."},
        {"role": "USER", "message": "What is the capital of France?"},
        {"role": "CHATBOT", "message": "Paris"},
    ]

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        sdk = cohere.Client(api_key="test", httpx_client=http_client)
        client = instructor.from_cohere(sdk, mode=mode)
        result = client.create(
            model="command-r-08-2024",
            response_model=Answer,
            messages=messages,
            max_retries=2,
        )

    assert result.answer == "London"
    first, retry = requests
    first_history = first["chat_history"]
    if mode == Mode.TOOLS:
        assert first_history[0]["role"] == "USER"
        first_history = first_history[1:]
    assert first_history == expected_history
    assert messages[-1]["content"] in first["message"]
    assert retry["chat_history"] == first["chat_history"] + [
        {"role": "USER", "message": first["message"]}
    ]
