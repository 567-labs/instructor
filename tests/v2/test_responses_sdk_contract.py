"""Responses wire contracts: real SDKs and loopback HTTP, no provider service."""

from __future__ import annotations

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from inspect import isawaitable
import threading
from typing import Any

import httpx
import openai
from openai.types.responses import Response, ResponseFunctionToolCall
from pydantic import BaseModel
import pytest

import instructor
from instructor.v2.core.client import AsyncInstructor
from instructor import Mode
from instructor.core.exceptions import InstructorRetryException


class Answer(BaseModel):
    value: int


@pytest.fixture
def responses_endpoint() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    calls: list[dict[str, Any]] = []
    response = Response(
        id="resp_local",
        created_at=1,
        model="local-contract",
        object="response",
        output=[
            ResponseFunctionToolCall(
                type="function_call",
                call_id="call_local",
                name="Answer",
                arguments='{"value":42}',
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"path": self.path, "body": request})
            bad_request = request["model"] == "reject"
            body = (
                json.dumps(
                    {
                        "error": {
                            "message": "invalid local model",
                            "type": "invalid_request_error",
                            "code": "model_not_found",
                        }
                    }
                )
                if bad_request
                else response.model_dump_json()
            ).encode()
            self.send_response(400 if bad_request else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("x-request-id", "req_local")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ARG002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def assert_response_contract(
    result: Any, calls: list[dict[str, Any]], raw: bool
) -> None:
    if raw:
        assert isinstance(result, Response)
        assert result.id == "resp_local"
        assert isinstance(result.output[0], ResponseFunctionToolCall)
    else:
        assert isinstance(result, Answer)
        assert result.value == 42
    assert len(calls) == 1
    assert calls[0]["path"] == "/v1/responses"
    body = calls[0]["body"]
    assert body["input"] == [{"role": "user", "content": "answer"}]
    assert "messages" not in body
    assert body["max_output_tokens"] == 100
    assert "max_tokens" not in body
    if raw:
        assert "tools" not in body
    else:
        assert body["tool_choice"] == {"type": "function", "name": "Answer"}
        assert (
            body["tools"][0]["parameters"]["properties"]["value"]["type"] == "integer"
        )
        assert "function" not in body["tools"][0]


@pytest.mark.parametrize("raw", [False, True])
def test_sync_responses_sdk_contract(responses_endpoint, raw: bool) -> None:
    url, calls = responses_endpoint
    with openai.OpenAI(
        api_key="local-only",
        base_url=url,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    ) as sdk:
        client = instructor.from_openai(sdk, mode=Mode.RESPONSES_TOOLS)
        result = client.responses.create(
            model="local-contract",
            messages=[{"role": "user", "content": "answer"}],
            response_model=None if raw else Answer,
            max_tokens=100,
            max_retries=1,
        )
    assert_response_contract(result, calls, raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [False, True])
async def test_async_responses_sdk_contract(responses_endpoint, raw: bool) -> None:
    url, calls = responses_endpoint
    async with openai.AsyncOpenAI(
        api_key="local-only",
        base_url=url,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False),
    ) as sdk:
        client = instructor.from_openai(sdk, mode=Mode.RESPONSES_TOOLS)
        assert isinstance(client, AsyncInstructor)
        pending = client.responses.create(
            model="local-contract",
            messages=[{"role": "user", "content": "answer"}],
            response_model=None if raw else Answer,
            max_tokens=100,
            max_retries=1,
        )
        assert isawaitable(pending)
        result = await pending
    assert_response_contract(result, calls, raw)


def assert_sdk_error(error: InstructorRetryException, calls) -> None:
    assert len(calls) == 1
    assert calls[0]["path"] == "/v1/responses"
    assert error.n_attempts == 1
    cause = error.__cause__
    assert isinstance(cause, openai.BadRequestError)
    assert cause.status_code == 400
    assert cause.request_id == "req_local"
    assert cause.code == "model_not_found"


def test_sync_responses_preserves_sdk_error(responses_endpoint) -> None:
    url, calls = responses_endpoint
    with openai.OpenAI(
        api_key="local-only",
        base_url=url,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    ) as sdk:
        client = instructor.from_openai(sdk, mode=Mode.RESPONSES_TOOLS)
        with pytest.raises(InstructorRetryException) as caught:
            client.responses.create(
                model="reject",
                messages=[{"role": "user", "content": "answer"}],
                response_model=Answer,
                max_retries=1,
            )
    assert_sdk_error(caught.value, calls)


@pytest.mark.asyncio
async def test_async_responses_preserves_sdk_error(responses_endpoint) -> None:
    url, calls = responses_endpoint
    async with openai.AsyncOpenAI(
        api_key="local-only",
        base_url=url,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False),
    ) as sdk:
        client = instructor.from_openai(sdk, mode=Mode.RESPONSES_TOOLS)
        assert isinstance(client, AsyncInstructor)
        with pytest.raises(InstructorRetryException) as caught:
            pending = client.responses.create(
                model="reject",
                messages=[{"role": "user", "content": "answer"}],
                response_model=Answer,
                max_retries=1,
            )
            assert isawaitable(pending)
            await pending
    assert_sdk_error(caught.value, calls)
