"""Responses wire contracts: real SDKs and loopback HTTP, no provider service."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
import json
from inspect import isawaitable
from typing import Any

from tests import openai_http as httpx
from tests._http import JSONHandler, serve
import openai
from openai.types.responses import Response, ResponseFunctionToolCall
from openai.types.responses.response import IncompleteDetails
from pydantic import BaseModel
import pytest

import instructor
from instructor.v2.core.client import AsyncInstructor
from instructor import Mode
from instructor.core.exceptions import (
    IncompleteOutputException,
    InstructorRetryException,
)
from instructor.cache import AutoCache


class Answer(BaseModel):
    value: int
    label: str = "default-label"


@pytest.fixture
def responses_endpoint(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    calls: list[dict[str, Any]] = []
    status, reason = getattr(request, "param", ("completed", None))
    response = Response(
        id="resp_local",
        created_at=1,
        model="local-contract",
        object="response",
        status=status,
        incomplete_details=IncompleteDetails(reason=reason)
        if status == "incomplete"
        else None,
        metadata={"trace": "local-contract"},
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

    class Handler(JSONHandler):
        def do_POST(self) -> None:
            request = self.read_json()
            calls.append({"path": self.path, "body": request})
            if request["model"] == "reject":
                error = {
                    "message": "invalid local model",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
                body, status = json.dumps({"error": error}).encode(), 400
            else:
                body, status = response.model_dump_json().encode(), 200
            self.send_json(body, status, {"x-request-id": "req_local"})

    with serve(Handler) as url:
        yield f"{url}/v1", calls


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


@asynccontextmanager
async def responses_client(
    url: str, is_async: bool
) -> AsyncGenerator[Callable[..., Awaitable[Any]], None]:
    """Yield an awaitable ``responses.create`` backed by a real sync or async SDK."""
    defaults: dict[str, Any] = dict(
        model="local-contract",
        messages=[{"role": "user", "content": "answer"}],
        response_model=Answer,
        max_retries=1,
    )
    sdk_options: dict[str, Any] = dict(
        api_key="local-only", base_url=url, max_retries=0
    )
    if not is_async:
        with openai.OpenAI(
            **sdk_options, http_client=httpx.Client(trust_env=False)
        ) as sdk:
            client = instructor.from_openai(sdk, mode=Mode.RESPONSES_TOOLS)

            async def create(**kwargs: Any) -> Any:
                return client.responses.create(**{**defaults, **kwargs})

            yield create
        return
    async with openai.AsyncOpenAI(
        **sdk_options, http_client=httpx.AsyncClient(trust_env=False)
    ) as async_sdk:
        async_client = instructor.from_openai(async_sdk, mode=Mode.RESPONSES_TOOLS)
        assert isinstance(async_client, AsyncInstructor)

        async def create_async(**kwargs: Any) -> Any:
            pending = async_client.responses.create(**{**defaults, **kwargs})
            assert isawaitable(pending)
            return await pending

        yield create_async


IS_ASYNC = pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])


@pytest.mark.asyncio
@IS_ASYNC
@pytest.mark.parametrize("raw", [False, True])
async def test_responses_sdk_contract(
    responses_endpoint, is_async: bool, raw: bool
) -> None:
    url, calls = responses_endpoint
    async with responses_client(url, is_async) as create:
        result = await create(response_model=None if raw else Answer, max_tokens=100)
    assert_response_contract(result, calls, raw)


@pytest.mark.asyncio
@IS_ASYNC
async def test_responses_preserves_sdk_error(
    responses_endpoint, is_async: bool
) -> None:
    url, calls = responses_endpoint
    async with responses_client(url, is_async) as create:
        with pytest.raises(InstructorRetryException) as caught:
            await create(model="reject")
    error = caught.value
    assert len(calls) == 1
    assert calls[0]["path"] == "/v1/responses"
    assert error.n_attempts == 1
    cause = error.__cause__
    assert isinstance(cause, openai.BadRequestError)
    assert cause.status_code == 400
    assert cause.request_id == "req_local"
    assert cause.code == "model_not_found"


@pytest.mark.asyncio
@IS_ASYNC
@pytest.mark.parametrize(
    "responses_endpoint,reason",
    [
        (("incomplete", "max_output_tokens"), "max_output_tokens"),
        (("incomplete", "content_filter"), "content_filter"),
        (("incomplete", None), None),
    ],
    indirect=["responses_endpoint"],
)
async def test_incomplete_responses_are_not_cached(
    responses_endpoint, is_async: bool, reason: str | None
) -> None:
    url, calls = responses_endpoint
    cache = AutoCache()
    async with responses_client(url, is_async) as create:
        for attempt in range(2):
            with pytest.raises(IncompleteOutputException) as caught:
                await create(cache=cache)
            response = caught.value.last_completion
            assert isinstance(response, Response)
            assert response.status == "incomplete"
            assert response.incomplete_details is not None
            assert response.incomplete_details.reason == reason
            assert response.metadata == {"trace": "local-contract"}
            assert isinstance(response.output[0], ResponseFunctionToolCall)
            assert response.output[0].arguments == '{"value":42}'
            assert "incomplete" in str(caught.value)
            if reason is not None:
                assert reason in str(caught.value)
            assert len(calls) == attempt + 1
            assert calls[-1]["path"] == "/v1/responses"


@pytest.mark.asyncio
@IS_ASYNC
async def test_completed_responses_keep_defaults_and_cache(
    responses_endpoint, is_async: bool
) -> None:
    url, calls = responses_endpoint
    cache = AutoCache()
    async with responses_client(url, is_async) as create:
        results = [await create(cache=cache) for _ in range(2)]
    assert len(calls) == 1
    for result in results:
        assert isinstance(result, Answer)
        assert result.model_dump() == {"value": 42, "label": "default-label"}
