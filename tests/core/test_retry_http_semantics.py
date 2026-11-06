"""Real SDK requests against loopback HTTP; no provider credentials or mocks."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import openai
import pytest
from pydantic import BaseModel
from tenacity import AsyncRetrying, Retrying, stop_after_attempt

import instructor
from instructor.core.exceptions import InstructorRetryException


class Answer(BaseModel):
    value: int


@dataclass
class Reply:
    status: int = 200
    content: str = '{"value": 7}'
    delay: float = 0
    retry_after: float = 0.001


class Endpoint:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.replies: list[Reply] = []
        self.requests: list[dict[str, Any]] = []
        self.entered = threading.Event()
        self.url = ""


@pytest.fixture(params=["openai", "anthropic"])
def endpoint(request: pytest.FixtureRequest) -> Iterator[Endpoint]:
    if request.param == "anthropic":
        pytest.importorskip("anthropic")
    endpoint = Endpoint(request.param)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(endpoint.requests)
            endpoint.requests.append(body)
            endpoint.entered.set()
            reply = endpoint.replies[min(index, len(endpoint.replies) - 1)]
            time.sleep(reply.delay)
            if reply.status != 200:
                payload = {"error": {"message": "local failure", "type": "api_error"}}
            elif endpoint.provider == "anthropic":
                payload = {
                    "id": "msg-local",
                    "type": "message",
                    "role": "assistant",
                    "model": "local",
                    "content": [{"type": "text", "text": reply.content}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            else:
                payload = {
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "local",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply.content},
                            "finish_reason": "stop",
                        }
                    ],
                }
            encoded = json.dumps(payload).encode()
            try:
                self.send_response(reply.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Retry-After", str(reply.retry_after))
                self.end_headers()
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when the client times out or is cancelled.

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    endpoint.url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def observe(client: Any) -> dict[str, list[Any]]:
    events: dict[str, list[Any]] = {}
    for name in (
        "completion:kwargs",
        "completion:response",
        "completion:error",
        "parse:error",
        "completion:last_attempt",
    ):
        events[name] = []

        def record(*args: Any, event: str = name, **kwargs: Any) -> None:
            events[event].append((args, kwargs))

        client.on(name, record)
    return events


def make_client(
    endpoint: Endpoint, asynchronous: bool, sdk_retries: int
) -> tuple[Any, Any]:
    if endpoint.provider == "anthropic":
        import anthropic

        sdk_type = anthropic.AsyncAnthropic if asynchronous else anthropic.Anthropic
        sdk = sdk_type(
            api_key="local-only", base_url=endpoint.url, max_retries=sdk_retries
        )
        return sdk, instructor.from_anthropic(sdk, mode=instructor.Mode.JSON)
    sdk_type = openai.AsyncOpenAI if asynchronous else openai.OpenAI
    sdk = sdk_type(api_key="local-only", base_url=endpoint.url, max_retries=sdk_retries)
    return sdk, instructor.from_openai(sdk, mode=instructor.Mode.JSON)


async def extract(
    endpoint: Endpoint,
    asynchronous: bool,
    sdk_retries: int,
    retries: Any,
    timeout: Any = 2.0,
) -> tuple[Any, dict[str, list[Any]], float]:
    sdk, client = make_client(endpoint, asynchronous, sdk_retries)
    events = observe(client)
    started = time.monotonic()
    try:
        kwargs = dict(
            model="local",
            max_tokens=32,
            response_model=Answer,
            messages=[{"role": "user", "content": "Extract seven"}],
            max_retries=retries,
            timeout=timeout,
        )
        try:
            result = (
                await client.create(**kwargs)
                if asynchronous
                else client.create(**kwargs)
            )
        except InstructorRetryException as exc:
            result = exc
        return result, events, time.monotonic() - started
    finally:
        if asynchronous:
            await sdk.close()
        else:
            sdk.close()


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "statuses,contents,sdk_retries,attempts,http_calls,error_type",
    [
        ([200], ['{"value":7}'], 2, 1, 1, None),
        ([429, 200], ["", '{"value":7}'], 2, 1, 2, None),
        ([500, 200], ["", '{"value":7}'], 2, 1, 2, None),
        ([429], [""], 2, 1, 3, openai.RateLimitError),
        ([500], [""], 0, 1, 1, openai.InternalServerError),
        ([400], [""], 2, 1, 1, openai.BadRequestError),
        ([200, 200], ["{", '{"value":7}'], 2, 2, 2, None),
        ([200], ['{"value":"bad"}'], 2, 3, 3, InstructorRetryException),
        ([429, 200, 500, 200], ["", "{", "", '{"value":7}'], 2, 2, 4, None),
    ],
    ids=[
        "success",
        "rate-recovery",
        "server-recovery",
        "sdk-exhaustion",
        "sdk-disabled",
        "bad-request",
        "malformed-recovery",
        "validation-exhaustion",
        "mixed-recovery",
    ],
)
def test_http_attempts(
    endpoint: Endpoint,
    asynchronous: bool,
    statuses: list[int],
    contents: list[str],
    sdk_retries: int,
    attempts: int,
    http_calls: int,
    error_type: Any,
) -> None:
    endpoint.replies = [
        Reply(status, content) for status, content in zip(statuses, contents)
    ]
    result, events, elapsed = asyncio.run(
        extract(endpoint, asynchronous, sdk_retries, retries=2)
    )
    assert len(endpoint.requests) == http_calls
    assert len(events["completion:kwargs"]) == attempts
    if error_type is None:
        assert result.model_dump() == {"value": 7}
        assert not events["completion:last_attempt"]
    else:
        assert isinstance(result, InstructorRetryException)
        assert result.n_attempts == attempts
        if error_type is not InstructorRetryException:
            assert type(result.__cause__).__name__ == error_type.__name__
            assert len(events["completion:error"]) == 1
        assert events["completion:last_attempt"][0][1]["attempt_number"] == attempts
    successful_http = sum(
        statuses[min(i, len(statuses) - 1)] == 200 for i in range(http_calls)
    )
    assert len(events["completion:response"]) == successful_http
    assert len(events["parse:error"]) == successful_http - (error_type is None)
    if statuses[0] == 429 and http_calls > 1:
        assert endpoint.requests[0] == endpoint.requests[1]
    if attempts > 1:
        assert len(endpoint.requests[-1]["messages"]) > len(
            endpoint.requests[0]["messages"]
        )
    print(
        f"provider={endpoint.provider} async={asynchronous} statuses={statuses} "
        f"instructor={attempts} http={http_calls} success={error_type is None} seconds={elapsed:.4f}"
    )


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("exhausted", [False, True], ids=["recovers", "exhausted"])
def test_custom_policy_can_retry_sdk_exhaustion(
    endpoint: Endpoint, asynchronous: bool, exhausted: bool
) -> None:
    endpoint.replies = [Reply(500), Reply(500), Reply(500) if exhausted else Reply()]
    policy = (AsyncRetrying if asynchronous else Retrying)(stop=stop_after_attempt(2))
    result, events, _ = asyncio.run(extract(endpoint, asynchronous, 1, policy))
    assert len(events["completion:kwargs"]) == 2
    if exhausted:
        assert isinstance(result, InstructorRetryException)
        assert result.n_attempts == 2
        assert len(endpoint.requests) == 4
        assert events["completion:last_attempt"][0][1]["is_last_attempt"] is True
    else:
        assert result.model_dump() == {"value": 7}
        assert len(endpoint.requests) == 3
        assert not events["completion:last_attempt"]
    assert all(
        event[1]["is_last_attempt"] is False for event in events["completion:error"]
    )
    assert all(event[1]["max_attempts"] is None for event in events["completion:error"])


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("valid", [False, True], ids=["invalid", "valid"])
def test_numeric_timeout_is_not_a_deadline(
    endpoint: Endpoint, asynchronous: bool, valid: bool
) -> None:
    endpoint.replies = [
        Reply(429, retry_after=0.15),
        Reply(content='{"value":7}' if valid else "{"),
    ]
    result, events, elapsed = asyncio.run(
        extract(endpoint, asynchronous, 1, 3, timeout=0.05)
    )
    assert elapsed >= 0.15
    assert len(endpoint.requests) == 2
    assert len(events["completion:kwargs"]) == 1
    if valid:
        assert result.model_dump() == {"value": 7}
    else:
        assert isinstance(result, InstructorRetryException)
        assert result.n_attempts == 1
        assert events["completion:last_attempt"][0][1]["is_last_attempt"] is True


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("sdk_retries", [0, 1])
def test_transport_read_timeout(
    endpoint: Endpoint, asynchronous: bool, sdk_retries: int
) -> None:
    endpoint.replies = [Reply(delay=0.3)]
    result, events, _ = asyncio.run(
        extract(endpoint, asynchronous, sdk_retries, 3, timeout=0.05)
    )
    assert type(result.__cause__).__name__ == "APITimeoutError"
    assert result.n_attempts == 1
    assert len(endpoint.requests) == sdk_retries + 1
    assert not events["completion:response"]


@pytest.mark.parametrize("deadline", [False, True], ids=["cancel", "wait-for"])
def test_async_cancellation_does_not_retry(endpoint: Endpoint, deadline: bool) -> None:
    endpoint.replies = [Reply(delay=0.5)]

    async def run() -> None:
        sdk, client = make_client(endpoint, True, 2)
        async with sdk:
            events = observe(client)
            task = asyncio.create_task(
                client.create(
                    model="local",
                    max_tokens=32,
                    response_model=Answer,
                    messages=[{"role": "user", "content": "Extract seven"}],
                    max_retries=3,
                )
            )
            assert await asyncio.to_thread(endpoint.entered.wait, 2)
            if deadline:
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(task, timeout=0.02)
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert task.cancelled()
            assert len(endpoint.requests) == 1
            assert len(events["completion:kwargs"]) == 1
            assert not events["completion:error"]
            assert not events["completion:last_attempt"]

    asyncio.run(run())
