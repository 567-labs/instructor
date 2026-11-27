from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

import instructor
from instructor import Mode
from instructor.core.hooks import Hooks
from instructor.v2.dsl.response_list import ListResponse


class Answer(BaseModel):
    value: int


CASES = [
    pytest.param(int, 7, 7, id="int"),
    pytest.param(str, "Ada", "Ada", id="str"),
    pytest.param(bool, False, False, id="bool"),
    pytest.param(float, 1.5, 1.5, id="float"),
    pytest.param(dict[str, int], {"a": 7}, {"a": 7}, id="dict"),
    pytest.param(tuple[int, str], [7, "Ada"], (7, "Ada"), id="tuple"),
    pytest.param(set[int], [7, 8], {7, 8}, id="set"),
    pytest.param(list[int], [7, 8], [7, 8], id="list"),
    pytest.param(Answer, {"value": 7}, Answer(value=7), id="model"),
]


def completion_transport(
    wire: Any, requests: list[dict[str, Any]]
) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        function = body["tools"][0]["function"]
        value = wire(len(requests)) if callable(wire) else wire
        arguments = value if function["name"] == "Answer" else {"content": value}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-native",
                "object": "chat.completion",
                "created": 1,
                "model": "local-contract",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-native",
                                    "type": "function",
                                    "function": {
                                        "name": function["name"],
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "total_tokens": 12,
                },
            },
        )

    return httpx.MockTransport(respond)


def assert_completion(result: Any, raw: Any, expected: Any) -> None:
    assert_value(result, expected)
    assert isinstance(raw, ChatCompletion)
    assert raw.id == "chatcmpl-native"
    assert raw.usage is not None
    assert raw.usage.total_tokens == 12
    if isinstance(expected, list):
        assert isinstance(result, ListResponse)
        assert result.get_raw_response() is raw


def assert_value(result: Any, expected: Any) -> None:
    if isinstance(expected, BaseModel):
        assert isinstance(result, type(expected))
        assert result.model_dump() == expected.model_dump()
    else:
        assert result == expected
        if not isinstance(expected, list):
            assert type(result) is type(expected)


@pytest.mark.parametrize(("annotation", "wire", "expected"), CASES)
def test_create_with_completion_native_sync(
    annotation: Any, wire: Any, expected: Any
) -> None:
    requests: list[dict[str, Any]] = []
    with httpx.Client(transport=completion_transport(wire, requests)) as http_client:
        client = instructor.from_provider(
            "openai/local-contract",
            api_key="local-only",
            http_client=http_client,
            mode=Mode.TOOLS,
        )
        result, raw = client.create_with_completion(
            response_model=annotation,
            messages=[{"role": "user", "content": "Extract a value"}],
            max_retries=0,
        )
        assert_completion(result, raw, expected)
        ordinary = client.create(
            response_model=annotation,
            messages=[{"role": "user", "content": "Extract a value"}],
            max_retries=0,
        )
        assert_value(ordinary, expected)
        assert requests[0] == requests[1]


def test_native_completion_preserves_retries_and_hooks() -> None:
    requests: list[dict[str, Any]] = []
    observed: list[ChatCompletion] = []
    hooks = Hooks()
    hooks.on("completion:response", observed.append)
    transport = completion_transport(
        lambda attempt: "invalid" if attempt == 1 else 7, requests
    )
    with httpx.Client(transport=transport) as http_client:
        client = instructor.from_provider(
            "openai/local-contract",
            api_key="local-only",
            http_client=http_client,
            mode=Mode.TOOLS,
        )
        result, raw = client.create_with_completion(
            response_model=int,
            messages=[{"role": "user", "content": "Extract"}],
            max_retries=1,
            hooks=hooks,
        )
    assert result == 7
    assert len(requests) == len(observed) == 2
    assert raw is observed[-1]
    assert raw.usage is not None
    assert raw.usage.total_tokens == 24


@pytest.mark.asyncio
async def test_native_completion_preserves_async_retries_and_hooks() -> None:
    requests: list[dict[str, Any]] = []
    observed: list[ChatCompletion] = []
    hooks = Hooks()
    hooks.on("completion:response", observed.append)
    transport = completion_transport(
        lambda attempt: "invalid" if attempt == 1 else 7, requests
    )
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = instructor.from_provider(
            "openai/local-contract",
            api_key="local-only",
            http_client=http_client,
            mode=Mode.TOOLS,
            async_client=True,
        )
        result, raw = await client.create_with_completion(
            response_model=int,
            messages=[{"role": "user", "content": "Extract"}],
            max_retries=1,
            hooks=hooks,
        )
    assert result == 7
    assert len(requests) == len(observed) == 2
    assert raw is observed[-1]
    assert raw.usage is not None
    assert raw.usage.total_tokens == 24


@pytest.mark.asyncio
@pytest.mark.parametrize(("annotation", "wire", "expected"), CASES)
async def test_create_with_completion_native_async(
    annotation: Any, wire: Any, expected: Any
) -> None:
    requests: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        transport=completion_transport(wire, requests)
    ) as http_client:
        client = instructor.from_provider(
            "openai/local-contract",
            api_key="local-only",
            http_client=http_client,
            mode=Mode.TOOLS,
            async_client=True,
        )
        result, raw = await client.create_with_completion(
            response_model=annotation,
            messages=[{"role": "user", "content": "Extract a value"}],
            max_retries=0,
        )
        assert_completion(result, raw, expected)
        ordinary = await client.create(
            response_model=annotation,
            messages=[{"role": "user", "content": "Extract a value"}],
            max_retries=0,
        )
        assert_value(ordinary, expected)
        assert requests[0] == requests[1]
