from __future__ import annotations

import json
from collections.abc import Callable, Generator
from copy import deepcopy
from typing import Any, cast

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator
from tenacity import AsyncRetrying, AttemptManager, Retrying

from instructor.cache import BaseCache
from instructor.v2.core.errors import (
    InstructorRetryException,
)
from instructor.v2.core.hooks import Hooks
from instructor.v2.core.mode import Mode
from instructor.v2.core.patch import patch, patch_v2
from instructor.v2.core.providers import Provider
from instructor.v2.core.retry import (
    _finalize_parsed_response,
    retry_async,
    retry_async_v2,
    retry_sync,
    retry_sync_v2,
)
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.response_list import ListResponse
from instructor.v2.dsl.simple_type import ModelAdapter
from tests.coverage._openai import FinishReason, chat_completion


class Answer(BaseModel):
    value: int


def _completion(value: Any, finish_reason: FinishReason = "stop") -> ChatCompletion:
    return chat_completion(
        content=json.dumps({"value": value}), finish_reason=finish_reason
    )


class RecordingCache(BaseCache):
    def __init__(self, fail_on_store: bool = False) -> None:
        self.values: dict[str, Any] = {}
        self.ttls: list[int | None] = []
        self.fail_on_store = fail_on_store

    def get(self, key: str) -> Any | None:
        return self.values.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self.ttls.append(ttl)
        if self.fail_on_store:
            raise ModuleNotFoundError("optional cache backend is unavailable")
        self.values[key] = value


class UnsupportedCache:
    def get(self, _key: str) -> Any:
        raise AssertionError("unsupported cache must not be read")

    def set(self, _key: str, _value: Any, _ttl: int | None = None) -> None:
        raise AssertionError("unsupported cache must not be written")


def test_patch_requires_a_target_and_supports_a_create_callable() -> None:
    patch_without_target = cast(Callable[[], object], patch)
    with pytest.raises(ValueError, match="Either client or create must be provided"):
        patch_without_target()

    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs)
        return _completion(8)

    patched = patch(create=create, provider=Provider.OPENAI, mode=Mode.JSON)
    result = patched(
        response_model=Answer,
        model="test-model",
        messages=[{"role": "user", "content": "give me eight"}],
    )

    assert result.value == 8
    assert calls[0]["model"] == "test-model"
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_finalize_converts_iterable_models_and_unwraps_simple_types() -> None:
    response = _completion(0)
    iterable_type = IterableModel(Answer)
    iterable = iterable_type.model_validate(
        {"tasks": [Answer(value=1), Answer(value=2)]}
    )

    finalized = _finalize_parsed_response(iterable, response)

    assert isinstance(finalized, ListResponse)
    assert list(finalized) == [Answer(value=1), Answer(value=2)]
    assert finalized.get_raw_response() is response

    adapter_type = cast(type[BaseModel], ModelAdapter[int])
    adapted = adapter_type.model_validate({"content": 19})
    assert _finalize_parsed_response(adapted, response) == 19


@pytest.mark.asyncio
async def test_retry_async_without_response_model_forwards_args_and_kwargs() -> None:
    calls: list[tuple[str, list[dict[str, str]], int]] = []
    raw_response = _completion(31)

    async def create(
        request_id: str, *, messages: list[dict[str, str]], timeout: int
    ) -> ChatCompletion:
        calls.append((request_id, messages, timeout))
        return raw_response

    messages = [{"role": "user", "content": "return raw response"}]
    result = await retry_async_v2(
        func=create,
        response_model=None,
        provider=Provider.OPENAI,
        mode=Mode.JSON,
        context=None,
        max_retries=1,
        args=("raw-request",),
        kwargs={"messages": messages, "timeout": 15},
        strict=True,
    )

    assert result is raw_response
    assert calls == [("raw-request", messages, 15)]


class NoAttemptsRetrying(Retrying):
    def __iter__(self) -> Generator[AttemptManager, None, None]:
        yield from ()


class NoAttemptsAsyncRetrying(AsyncRetrying):
    def __aiter__(self) -> NoAttemptsAsyncRetrying:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


sync_and_async = pytest.mark.parametrize(
    "is_async", [False, True], ids=["sync", "async"]
)


def replaying(replies: list[Any], is_async: bool) -> tuple[Any, list[Any]]:
    """Return a create function replaying `replies` and its recorded calls."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def reply(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        calls.append((args, kwargs))
        result = replies[min(len(calls), len(replies)) - 1]
        if isinstance(result, Exception):
            raise result
        return result

    if is_async:

        async def acreate(*args: Any, **kwargs: Any) -> Any:
            return reply(args, kwargs)

        return acreate, calls
    return lambda *args, **kwargs: reply(args, kwargs), calls


async def resolve(value: Any, is_async: bool) -> Any:
    return await value if is_async else value


CACHE_CASES = {
    # (cache, cache_ttl, recorded ttls, provider calls for two equal requests)
    "store-then-hit": (RecordingCache, 45, [45], 1),
    "missing-backend-invalid-ttl": (
        lambda: RecordingCache(fail_on_store=True),
        "not-an-integer",
        [None, None],
        2,
    ),
    "missing-backend": (lambda: RecordingCache(fail_on_store=True), 10, [10, 10], 2),
    "unsupported-cache": (UnsupportedCache, 30, None, 2),
}


@pytest.mark.asyncio
@sync_and_async
@pytest.mark.parametrize(
    ("make_cache", "ttl", "ttls", "n_calls"), CACHE_CASES.values(), ids=CACHE_CASES
)
async def test_patch_cache_policy(
    make_cache: Callable[[], Any], ttl: Any, ttls: Any, n_calls: int, is_async: bool
) -> None:
    create, calls = replaying([_completion(13)], is_async)
    cache = make_cache()
    patched = patch_v2(create, Provider.OPENAI, Mode.JSON, default_model="default")

    for _ in range(2):
        result = patched(
            response_model=Answer,
            messages=[{"role": "user", "content": "thirteen"}],
            cache=cache,
            cache_ttl=ttl,
        )
        assert (await resolve(result, is_async)).value == 13

    assert len(calls) == n_calls
    assert calls[0][1]["model"] == "default"
    assert "cache" not in calls[0][1]
    assert "cache_ttl" not in calls[0][1]
    if ttls is not None:
        assert cache.ttls == ttls


@pytest.mark.asyncio
@sync_and_async
async def test_retry_wrapper_reasks_and_forwards_args_timeout_stream(
    is_async: bool,
) -> None:
    create, calls = replaying([_completion("invalid"), _completion("29")], is_async)
    parse_errors: list[dict[str, Any]] = []
    hooks = Hooks()
    hooks.on("parse:error", lambda _error, **metadata: parse_errors.append(metadata))
    retry: Any = retry_async if is_async else retry_sync

    result = await resolve(
        retry(
            func=create,
            response_model=Answer,
            args=("request-id",),
            kwargs={
                "messages": [{"role": "user", "content": "twenty-nine"}],
                "timeout": 30.0,
                "stream": True,
            },
            max_retries=1,
            strict=False,
            mode=Mode.JSON,
            provider=Provider.OPENAI,
            hooks=hooks,
        ),
        is_async,
    )

    assert result.value == 29
    assert len(calls) == 2
    assert calls[0][0] == ("request-id",)
    assert calls[0][1]["timeout"] == 30.0
    assert calls[0][1]["stream"] is True
    assert calls[1][1]["messages"][-1]["role"] == "user"
    assert "Correct your JSON ONLY RESPONSE" in calls[1][1]["messages"][-1]["content"]
    assert parse_errors == [
        {"attempt_number": 1, "max_attempts": 2, "is_last_attempt": False}
    ]


@pytest.mark.asyncio
@sync_and_async
async def test_retry_wrapper_defaults_are_strict_and_allow_one_retry(
    is_async: bool,
) -> None:
    contexts: list[Any] = []
    completion_kwargs: list[dict[str, Any]] = []
    responses: list[Any] = []
    parse_errors: list[tuple[Exception, dict[str, Any]]] = []
    last_attempts: list[tuple[Exception, dict[str, Any]]] = []

    class ContextAnswer(BaseModel):
        value: int

        @field_validator("value", mode="before")
        @classmethod
        def record_context(cls, value: Any, info: ValidationInfo) -> Any:
            contexts.append(info.context)
            return value

    create, calls = replaying([_completion("41")], is_async)
    hooks = Hooks()
    hooks.on(
        "completion:kwargs",
        lambda **kwargs: completion_kwargs.append(deepcopy(kwargs)),
    )
    hooks.on("completion:response", responses.append)
    hooks.on(
        "parse:error",
        lambda error, **metadata: parse_errors.append((error, metadata)),
    )
    hooks.on(
        "completion:last_attempt",
        lambda error, **metadata: last_attempts.append((error, metadata)),
    )
    context = {"tenant": "strict"}
    retry: Any = retry_async if is_async else retry_sync

    with pytest.raises(InstructorRetryException) as exc_info:
        await resolve(
            retry(
                func=create,
                response_model=ContextAnswer,
                args=("request",),
                kwargs={"messages": [{"role": "user", "content": "strict value"}]},
                context=context,
                mode=Mode.JSON,
                provider=Provider.OPENAI,
                hooks=hooks,
            ),
            is_async,
        )

    assert len(calls) == 2
    assert all(args == ("request",) for args, _ in calls)
    assert contexts == [context, context]
    assert all(seen is context for seen in contexts)
    assert len(completion_kwargs) == 2
    assert completion_kwargs[0]["messages"] == [
        {"role": "user", "content": "strict value"}
    ]
    assert (
        "Correct your JSON ONLY RESPONSE"
        in completion_kwargs[1]["messages"][-1]["content"]
    )
    assert len(responses) == 2
    assert all(isinstance(error, ValidationError) for error, _ in parse_errors)
    assert all("valid integer" in str(error) for error, _ in parse_errors)
    last = {"attempt_number": 2, "max_attempts": 2, "is_last_attempt": True}
    assert [metadata for _, metadata in parse_errors] == [
        {"attempt_number": 1, "max_attempts": 2, "is_last_attempt": False},
        last,
    ]
    assert [(type(error), metadata) for error, metadata in last_attempts] == [
        (ValidationError, last)
    ]
    assert exc_info.value.n_attempts == 2
    assert len(exc_info.value.failed_attempts or []) == 2


@pytest.mark.asyncio
@sync_and_async
async def test_retry_wraps_an_api_error_without_hooks_or_retries(
    is_async: bool,
) -> None:
    create, calls = replaying([ConnectionError("provider is unavailable")], is_async)
    messages = [{"role": "user", "content": "answer once"}]
    retry: Any = retry_async_v2 if is_async else retry_sync_v2

    with pytest.raises(
        InstructorRetryException, match="provider is unavailable"
    ) as exc:
        await resolve(
            retry(
                func=create,
                response_model=Answer,
                provider=Provider.OPENAI,
                mode=Mode.JSON,
                context=None,
                max_retries=3,
                args=("request-id",),
                kwargs={"messages": messages, "timeout": 15},
                strict=True,
                hooks=None,
            ),
            is_async,
        )

    assert calls == [(("request-id",), {"messages": messages, "timeout": 15})]
    assert isinstance(exc.value.__cause__, ConnectionError)
    assert exc.value.n_attempts == 1
    assert exc.value.failed_attempts == []
    assert exc.value.last_completion is None
    assert exc.value.messages == messages
    assert exc.value.create_kwargs == {"messages": messages, "timeout": 15}


class NoAttemptsRetrying(Retrying):
    def __iter__(self) -> Generator[AttemptManager, None, None]:
        yield from ()


class NoAttemptsAsyncRetrying(AsyncRetrying):
    def __aiter__(self) -> NoAttemptsAsyncRetrying:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


@pytest.mark.asyncio
@sync_and_async
async def test_retry_reports_a_policy_that_yields_no_attempts(is_async: bool) -> None:
    create, calls = replaying([_completion(31)], is_async)
    retry: Any = retry_async_v2 if is_async else retry_sync_v2
    policy = NoAttemptsAsyncRetrying() if is_async else NoAttemptsRetrying()

    with pytest.raises(InstructorRetryException, match="Unknown error") as exc_info:
        await resolve(
            retry(
                func=create,
                response_model=Answer,
                provider=Provider.OPENAI,
                mode=Mode.JSON,
                context=None,
                max_retries=policy,
                args=(),
                kwargs={"messages": [{"role": "user", "content": "unused"}]},
                strict=True,
            ),
            is_async,
        )

    assert exc_info.value.n_attempts == 0
    assert exc_info.value.create_kwargs == {
        "messages": [{"role": "user", "content": "unused"}]
    }
    assert calls == []
