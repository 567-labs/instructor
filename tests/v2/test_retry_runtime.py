from __future__ import annotations

import json
import logging
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, ValidationError, field_validator
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
)

from instructor import Mode, Provider
from instructor.v2.core.errors import InstructorRetryException, ResponseParsingError
from instructor.v2.core.hooks import Hooks
from instructor.v2.core.retry import (
    _finalize_parsed_response,
    _initialize_usage,
    retry_async_v2,
    retry_sync_v2,
)
from instructor.v2.dsl.partial import Partial
from instructor.v2.dsl.response_list import ListResponse
from tests.coverage._openai import chat_completion


class Answer(BaseModel):
    value: int


def _validation_error(value: str = "bad") -> ValidationError:
    try:
        Answer.model_validate({"value": value})
    except ValidationError as exc:
        return exc
    raise AssertionError("Expected a validation error")


def test_finalize_parsed_response_wraps_plain_list_and_sets_raw_response() -> None:
    response = object()
    parsed = [Answer(value=1), Answer(value=2)]

    finalized = _finalize_parsed_response(parsed, response)

    assert isinstance(finalized, ListResponse)
    assert list(finalized) == parsed
    assert finalized._raw_response is response


def test_initialize_usage_returns_openai_usage_shape() -> None:
    usage = _initialize_usage(Provider.OPENAI)

    assert usage.completion_tokens == 0
    assert usage.prompt_tokens == 0
    assert usage.total_tokens == 0


def test_retry_sync_v2_returns_raw_result_when_no_response_model() -> None:
    def fake_func(*args: Any, **kwargs: Any) -> str:
        return f"{args[0]}:{kwargs['suffix']}"

    result = retry_sync_v2(
        func=fake_func,
        response_model=None,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=1,
        args=("hello",),
        kwargs={"suffix": "world"},
        strict=True,
        hooks=None,
    )

    assert result == "hello:world"


# --- real-handler retry loop, sync and async --------------------------------


def _completion(value: Any) -> ChatCompletion:
    return chat_completion(content=json.dumps({"value": value}))


def _anthropic_text(text: str) -> Any:
    types = pytest.importorskip("anthropic.types")
    return types.Message.model_validate(
        {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


class Recorder:
    """Plays scripted provider replies and records every hook emission."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []
        self.events: dict[str, list[Any]] = {
            name: []
            for name in (
                "completion:kwargs",
                "completion:response",
                "parse:error",
                "completion:error",
                "completion:last_attempt",
            )
        }
        self.hooks = Hooks()
        for hook, events in self.events.items():
            name = cast(Any, hook)
            if name == "completion:response":
                self.hooks.on(name, events.append)
            elif name == "completion:kwargs":
                self.hooks.on(name, lambda events=events, **kw: events.append(kw))
            else:
                self.hooks.on(
                    name,
                    lambda error, events=events, **meta: events.append((error, meta)),
                )

    def reply(self, kwargs: dict[str, Any]) -> Any:
        # Snapshot: the retry loop extends the same messages list in place.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        reply = self.replies[len(self.calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def run(
        self,
        is_async: bool,
        max_retries: int | str = 1,
        *,
        provider: Provider = Provider.OPENAI,
        mode: Mode = Mode.JSON,
        strict: bool = True,
    ) -> Any:
        policy: Any = max_retries
        if max_retries == "tenacity":
            policy = (AsyncRetrying if is_async else Retrying)(
                stop=stop_after_attempt(2),
                retry=retry_if_exception_type(ValidationError),
                reraise=True,
            )
        options: dict[str, Any] = {
            "response_model": Answer,
            "provider": provider,
            "mode": mode,
            "context": {"tenant": "acme"},
            "max_retries": policy,
            "args": (),
            "kwargs": {"messages": [{"role": "user", "content": "first"}]},
            "strict": strict,
            "hooks": self.hooks,
        }
        if is_async:

            async def acreate(**kwargs: Any) -> Any:
                return self.reply(kwargs)

            return await retry_async_v2(func=acreate, **options)
        return retry_sync_v2(func=lambda **kwargs: self.reply(kwargs), **options)


sync_and_async = pytest.mark.parametrize(
    "is_async", [False, True], ids=["sync", "async"]
)
LAST = {"attempt_number": 2, "max_attempts": 2, "is_last_attempt": True}


@pytest.mark.asyncio
@sync_and_async
@pytest.mark.parametrize("policy", [1, "tenacity"])
async def test_retry_reasks_with_feedback_until_valid(
    is_async: bool, policy: int | str
) -> None:
    recorder = Recorder([_completion("bad"), _completion(7)])

    result = await recorder.run(is_async, policy)

    assert result == Answer(value=7)
    first, second = (call["messages"] for call in recorder.calls)
    assert len(second) > len(first)
    assert second[-1]["role"] == "user"
    events = recorder.events
    assert len(events["completion:kwargs"]) == len(events["completion:response"]) == 2
    [(error, meta)] = events["parse:error"]
    assert isinstance(error, ValidationError)
    assert meta["attempt_number"] == 1
    assert events["completion:error"] == events["completion:last_attempt"] == []


@pytest.mark.asyncio
@sync_and_async
@pytest.mark.parametrize("error", ["validation", "response-parsing", "json-decode"])
async def test_retry_retries_every_parse_error(is_async: bool, error: str) -> None:
    if error == "json-decode":
        # Non-strict Anthropic JSON decodes with json.loads.
        replies = [_anthropic_text("{bad"), _anthropic_text('{"value": 42}')]
        recorder = Recorder(replies)
        result = await recorder.run(
            is_async, provider=Provider.ANTHROPIC, mode=Mode.JSON, strict=False
        )
        expected: type[Exception] = json.JSONDecodeError
    else:
        bad = (
            _completion("bad")
            if error == "validation"
            else ChatCompletion(
                id="c", choices=[], created=0, model="m", object="chat.completion"
            )
        )
        recorder = Recorder([bad, _completion(42)])
        result = await recorder.run(is_async)
        expected = ValidationError if error == "validation" else ResponseParsingError

    assert result == Answer(value=42)
    assert len(recorder.calls) == 2
    assert isinstance(recorder.events["parse:error"][0][0], expected)


@pytest.mark.asyncio
@sync_and_async
@pytest.mark.parametrize("policy", [1, "tenacity"])
async def test_retry_exhaustion_reports_last_attempt(
    is_async: bool, policy: int | str
) -> None:
    replies = [_completion("bad"), _completion("worse")]
    recorder = Recorder(replies)

    with pytest.raises(InstructorRetryException) as exc_info:
        await recorder.run(is_async, policy)

    error = exc_info.value
    assert error.n_attempts == 2
    assert error.last_completion is replies[-1]
    assert error.create_kwargs is not None
    sent = recorder.calls[-1]["messages"]
    # The final kwargs keep the last request plus the final feedback turn.
    assert error.create_kwargs["messages"][: len(sent)] == sent
    assert error.create_kwargs["messages"][-1]["role"] == "user"
    assert len(error.failed_attempts or []) == 2
    assert recorder.events["completion:error"] == []
    [(_, meta)] = recorder.events["completion:last_attempt"]
    # A custom tenacity policy has no known attempt budget.
    assert meta == {**LAST, "max_attempts": 2 if policy == 1 else None}


@pytest.mark.asyncio
@sync_and_async
async def test_retry_api_error_is_terminal_last_attempt(is_async: bool) -> None:
    recorder = Recorder([RuntimeError("boom")])

    with pytest.raises(InstructorRetryException) as exc_info:
        await recorder.run(is_async, 3)

    assert len(recorder.calls) == exc_info.value.n_attempts == 1
    terminal = {"attempt_number": 1, "max_attempts": 4, "is_last_attempt": True}
    assert [meta for _, meta in recorder.events["completion:error"]] == [terminal]
    assert [meta for _, meta in recorder.events["completion:last_attempt"]] == [
        terminal
    ]


@pytest.mark.asyncio
@sync_and_async
async def test_retry_reports_terminal_api_error_after_validation_failure(
    is_async: bool,
) -> None:
    recorder = Recorder([_completion("bad"), RuntimeError("terminal API failure")])

    with pytest.raises(InstructorRetryException, match="terminal API failure") as exc:
        await recorder.run(is_async)

    assert exc.value.n_attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream_kwargs", "expected_stream"),
    [({}, False), ({"stream": True}, True)],
    ids=["default-stream", "explicit-stream"],
)
async def test_retry_async_v2_preserves_parser_response_and_usage_contract(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stream_kwargs: dict[str, bool],
    expected_stream: bool,
) -> None:
    response = SimpleNamespace(payload="answer")
    context = {"tenant": "async"}
    usage = {"tokens": 0}
    seen_responses: list[Any] = []
    seen_updates: list[tuple[Any, Any]] = []

    def parser(**kwargs: Any) -> Answer:
        assert kwargs == {
            "response": response,
            "response_model": Answer,
            "validation_context": context,
            "strict": False,
            "stream": expected_stream,
            "is_async": True,
        }
        return Answer(value=11)

    def initialize_usage(provider: Provider) -> dict[str, int]:
        assert provider is Provider.OPENAI
        return usage

    def update_usage(response: Any, total_usage: Any) -> Any:
        seen_updates.append((response, total_usage))
        return response

    hooks = Hooks()
    hooks.on("completion:response", lambda value: seen_responses.append(value))
    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=parser,
            reask_handler=lambda **call: call["kwargs"],
        ),
    )
    monkeypatch.setattr("instructor.v2.core.retry._initialize_usage", initialize_usage)
    monkeypatch.setattr("instructor.v2.core.retry.update_total_usage", update_usage)
    caplog.set_level(logging.DEBUG, logger="instructor.v2.retry")

    async def create(**_kwargs: Any) -> SimpleNamespace:
        return response

    result = await retry_async_v2(
        func=create,
        response_model=Answer,
        provider=Provider.OPENAI,
        mode=Mode.JSON,
        context=context,
        max_retries=0,
        args=(),
        kwargs={"messages": [], **stream_kwargs},
        strict=False,
        hooks=hooks,
    )

    assert result == Answer(value=11)
    assert getattr(result, "_raw_response", None) is response
    assert getattr(result, "_total_usage", None) is None
    assert seen_responses == [response]
    assert seen_updates == [(response, usage)]
    assert "Successfully parsed response on attempt 1" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_retries", "timeout"),
    [(0, None), (3, 0)],
    ids=["zero-retries", "expired-timeout"],
)
async def test_retry_async_v2_stops_after_one_attempt_when_retry_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    max_retries: int,
    timeout: int | None,
) -> None:
    calls = 0
    response = {"payload": "invalid"}

    async def create(**_kwargs: Any) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return response

    def retrying(*, stop: Any, retry: Any, reraise: bool) -> AsyncRetrying:
        return AsyncRetrying(stop=stop, retry=retry, reraise=reraise)

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=lambda **_kwargs: Answer.model_validate({"value": "bad"}),
            reask_handler=lambda **call: call["kwargs"],
        ),
    )
    monkeypatch.setattr("instructor.v2.core.retry.AsyncRetrying", retrying)
    caplog.set_level(logging.DEBUG, logger="instructor.v2.retry")

    with pytest.raises(InstructorRetryException) as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=max_retries,
            args=(),
            kwargs={"messages": [], "timeout": timeout},
            strict=True,
            hooks=None,
        )

    assert calls == 1
    assert exc_info.value.n_attempts == 1
    assert exc_info.value.last_completion is response
    assert "Validation error on attempt 1" in caplog.text
    assert "Max retries exceeded. Total attempts: 1" in caplog.text


@pytest.mark.asyncio
async def test_retry_async_v2_reports_each_validation_api_error_and_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    errors = [_validation_error(), _validation_error()]
    completion_errors: list[tuple[Exception, dict[str, Any]]] = []
    last_attempts: list[tuple[Exception, dict[str, Any]]] = []
    calls = 0

    async def create(**_kwargs: Any) -> dict[str, str]:
        nonlocal calls
        error = errors[calls]
        calls += 1
        raise error

    hooks = Hooks()
    hooks.on(
        "completion:error",
        lambda error, **metadata: completion_errors.append((error, metadata)),
    )
    hooks.on(
        "completion:last_attempt",
        lambda error, **metadata: last_attempts.append((error, metadata)),
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=lambda **_kwargs: Answer(value=1),
            reask_handler=lambda **call: call["kwargs"],
        ),
    )
    caplog.set_level(logging.ERROR, logger="instructor.v2.retry")

    with pytest.raises(InstructorRetryException) as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=1,
            args=(),
            kwargs={"messages": [{"role": "user", "content": "answer"}]},
            strict=True,
            hooks=hooks,
        )

    assert calls == 2
    assert [error for error, _ in completion_errors] == errors
    assert [metadata for _, metadata in completion_errors] == [
        {"attempt_number": 1, "max_attempts": 2, "is_last_attempt": False},
        {"attempt_number": 2, "max_attempts": 2, "is_last_attempt": True},
    ]
    assert last_attempts == [
        (
            errors[-1],
            {"attempt_number": 2, "max_attempts": 2, "is_last_attempt": True},
        )
    ]
    assert exc_info.value.n_attempts == 2
    assert exc_info.value.failed_attempts == []
    assert (
        f"API call failed on attempt 1 ({type(errors[0]).__name__})" in caplog.messages
    )
    assert (
        f"API call failed on attempt 2 ({type(errors[1]).__name__})" in caplog.messages
    )
    assert (
        f"Max retries exceeded. Total attempts: 2 ({type(errors[-1]).__name__})"
        in caplog.messages
    )
    assert str(errors[0]) not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callback", [False, True], ids=["reraise", "return-from-policy"]
)
async def test_retry_async_v2_keeps_every_failed_completion_on_exhaustion(
    callback: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses = [
        {"payload": "first"},
        {"payload": "second"},
        {"payload": "third"},
    ]
    errors = [_validation_error(f"bad-{n}") for n in range(3)]
    parse_errors: list[tuple[Exception, dict[str, Any]]] = []
    completion_responses: list[Any] = []
    updates: list[tuple[Any, Any]] = []
    usage = {"tokens": 7}
    calls = 0
    parser_calls = 0

    async def create(**_kwargs: Any) -> dict[str, str]:
        nonlocal calls
        response = responses[calls]
        calls += 1
        return response

    def parser(**_kwargs: Any) -> Answer:
        nonlocal parser_calls
        error = errors[parser_calls]
        parser_calls += 1
        raise error

    def reask(
        kwargs: dict[str, Any], response: Any, exception: ValidationError
    ) -> dict[str, Any]:
        assert response is responses[parser_calls - 1]
        assert exception is errors[parser_calls - 1]
        return {
            **kwargs,
            "messages": [
                *kwargs["messages"],
                {"role": "user", "content": f"retry-{parser_calls}"},
            ],
        }

    hooks = Hooks()
    hooks.on(
        "parse:error",
        lambda error, **metadata: parse_errors.append((error, metadata)),
    )
    hooks.on(
        "completion:response",
        lambda response: completion_responses.append(response),
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=parser,
            reask_handler=reask,
        ),
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry._initialize_usage",
        lambda provider: usage if provider is Provider.OPENAI else None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.update_total_usage",
        lambda response, total_usage: updates.append((response, total_usage)),
    )
    caplog.set_level(logging.DEBUG, logger="instructor.v2.retry")

    policy = (
        AsyncRetrying(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(ValidationError),
            retry_error_callback=lambda _state: None,
        )
        if callback
        else 2
    )
    with pytest.raises(InstructorRetryException) as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=policy,
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=hooks,
        )

    error = exc_info.value
    assert calls == parser_calls == 3
    assert completion_responses == responses
    assert updates == [(response, usage) for response in responses]
    assert [caught for caught, _ in parse_errors] == errors
    assert [metadata for _, metadata in parse_errors] == [
        {
            "attempt_number": attempt,
            "max_attempts": None if callback else 3,
            "is_last_attempt": attempt == 3 and not callback,
        }
        for attempt in (1, 2, 3)
    ]
    assert error.args[0] == str(errors[-1])
    assert error.n_attempts == 3
    assert error.last_completion is responses[-1]
    assert error.total_usage is usage
    assert error.messages == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "retry-1"},
        {"role": "user", "content": "retry-2"},
        {"role": "user", "content": "retry-3"},
    ]
    assert error.create_kwargs is not None
    assert error.create_kwargs["messages"] == error.messages
    assert error.failed_attempts is not None
    assert len(error.failed_attempts) == 3
    assert [attempt.attempt_number for attempt in error.failed_attempts] == [1, 2, 3]
    assert [attempt.exception for attempt in error.failed_attempts] == errors
    assert [attempt.completion for attempt in error.failed_attempts] == responses
    assert (
        f"Validation error on attempt 1 ({type(errors[0]).__name__})" in caplog.messages
    )
    assert (
        f"Validation error on attempt 2 ({type(errors[1]).__name__})" in caplog.messages
    )
    assert (
        f"Validation error on attempt 3 ({type(errors[2]).__name__})" in caplog.messages
    )
    if not callback:
        assert (
            f"Max retries exceeded. Total attempts: 3 ({type(errors[-1]).__name__})"
            in caplog.messages
        )


class NoAttemptsAsyncRetrying(AsyncRetrying):
    def __aiter__(self) -> NoAttemptsAsyncRetrying:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_retry_async_v2_reports_an_empty_retry_policy_without_losing_context(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    usage = {"tokens": 0}
    messages = [{"role": "user", "content": "unused"}]

    async def create(**_kwargs: Any) -> dict[str, str]:
        raise AssertionError("an empty retry policy must not call the provider")

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=lambda **_kwargs: Answer(value=1),
            reask_handler=lambda **call: call["kwargs"],
        ),
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry._initialize_usage",
        lambda provider: usage if provider is Provider.OPENAI else None,
    )
    caplog.set_level(logging.ERROR, logger="instructor.v2.retry")

    with pytest.raises(InstructorRetryException, match=r"^Unknown error$") as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=NoAttemptsAsyncRetrying(),
            args=(),
            kwargs={"messages": messages},
            strict=True,
            hooks=None,
        )

    error = exc_info.value
    assert error.n_attempts == 0
    assert error.last_completion is None
    assert error.total_usage is usage
    assert error.messages == messages
    assert error.create_kwargs == {"messages": messages}
    assert error.failed_attempts == []
    assert "Unexpected code path in retry_async_v2" in [
        record.getMessage() for record in caplog.records
    ]


def test_streaming_retry_exhaustion_preserves_validation_error() -> None:
    import httpx
    from openai import OpenAI

    import instructor

    class ImpossibleModel(BaseModel):
        value: str

        @field_validator("value")
        @classmethod
        def always_fail(cls, _value: str) -> str:
            raise ValueError("controlled invalid value")

    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        call = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {
                "name": body["tools"][0]["function"]["name"],
                "arguments": '{"value":"test"}',
            },
        }
        chunk = {
            "id": "chatcmpl-local",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "local",
            "choices": [{"index": 0, "delta": {"tool_calls": [call]}}],
        }
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    transport = httpx.MockTransport(respond)
    with OpenAI(api_key="local", http_client=httpx.Client(transport=transport)) as sdk:
        client = instructor.from_openai(sdk)
        with pytest.raises(InstructorRetryException) as caught:
            list(
                client.chat.completions.create_partial(
                    model="local",
                    max_retries=2,
                    messages=[{"role": "user", "content": "test"}],
                    response_model=ImpossibleModel,
                )
            )
    assert len(requests) == caught.value.n_attempts == 3
    assert isinstance(caught.value.__cause__, ValidationError)
    assert "controlled invalid value" in str(caught.value.__cause__)


@pytest.mark.parametrize("outcome", ["empty", "close", "complete", "invalid-final"])
def test_sync_partial_retry_stream_lifecycle(outcome: str) -> None:
    closed = False
    calls = 0

    def stream() -> Generator[ChatCompletionChunk, None, None]:
        nonlocal closed
        try:
            if outcome == "empty":
                return
            final = '"wrong"}' if outcome == "invalid-final" else "7}"
            for content in ('{"value":', final):
                yield ChatCompletionChunk.model_validate(
                    {
                        "id": "c",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "m",
                        "choices": [{"index": 0, "delta": {"content": content}}],
                    }
                )
        finally:
            closed = True

    def create(**_kwargs: Any) -> Generator[ChatCompletionChunk, None, None]:
        nonlocal calls
        calls += 1
        return stream()

    result = retry_sync_v2(
        create,
        Partial[Answer],
        Provider.OPENAI,
        Mode.JSON_SCHEMA,
        None,
        2,
        (),
        {"stream": True},
        True,
    )
    assert isinstance(result, Generator)
    if outcome == "close":
        result.close()
    elif outcome == "invalid-final":
        assert next(result).value is None
        with pytest.raises(ValidationError):
            next(result)
    else:
        assert list(result) == (
            []
            if outcome == "empty"
            else [Answer.model_construct(value=None), Answer(value=7)]
        )
    assert closed and calls == 1
