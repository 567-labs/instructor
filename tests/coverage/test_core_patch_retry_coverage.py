from __future__ import annotations

import json
from collections.abc import Callable, Generator
from copy import deepcopy
from typing import Any, cast

import pytest
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator
from tenacity import AsyncRetrying, AttemptManager, Retrying

from instructor.cache import BaseCache
from instructor.v2.core.errors import (
    IncompleteOutputException,
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
from instructor.v2.core.usage import _accumulate_models, update_total_usage
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


def test_openai_usage_adds_token_details_and_copies_totals_to_response() -> None:
    total_usage = CompletionUsage(
        completion_tokens=4,
        prompt_tokens=6,
        total_tokens=10,
        completion_tokens_details=CompletionTokensDetails(
            accepted_prediction_tokens=3,
            audio_tokens=1,
            reasoning_tokens=2,
            rejected_prediction_tokens=4,
        ),
        prompt_tokens_details=PromptTokensDetails.model_validate(
            {
                "audio_tokens": 3,
                "cached_tokens": 4,
                "cache_write_tokens": 5,
                "vendor_write_tokens": 2,
            }
        ),
    )
    response_usage = CompletionUsage(
        completion_tokens=7,
        prompt_tokens=11,
        total_tokens=18,
        completion_tokens_details=CompletionTokensDetails(
            accepted_prediction_tokens=7,
            audio_tokens=5,
            reasoning_tokens=8,
            rejected_prediction_tokens=9,
        ),
        prompt_tokens_details=PromptTokensDetails.model_validate(
            {
                "audio_tokens": 13,
                "cached_tokens": 21,
                "cache_write_tokens": 23,
                "vendor_write_tokens": 7,
            }
        ),
    )
    response = _completion(1)
    response.usage = response_usage

    updated = update_total_usage(response, total_usage)

    assert updated is response
    assert total_usage.completion_tokens == 11
    assert total_usage.prompt_tokens == 17
    assert total_usage.total_tokens == 28
    assert response_usage.completion_tokens == 11
    assert response_usage.prompt_tokens == 17
    assert response_usage.total_tokens == 28
    total_completion_details = total_usage.completion_tokens_details
    total_prompt_details = total_usage.prompt_tokens_details
    response_completion_details = response_usage.completion_tokens_details
    response_prompt_details = response_usage.prompt_tokens_details
    assert total_completion_details is not None
    assert total_prompt_details is not None
    assert response_completion_details is not None
    assert response_prompt_details is not None
    assert total_completion_details.accepted_prediction_tokens == 10
    assert total_completion_details.audio_tokens == 6
    assert total_completion_details.reasoning_tokens == 10
    assert total_completion_details.rejected_prediction_tokens == 13
    assert total_prompt_details.audio_tokens == 16
    assert total_prompt_details.cached_tokens == 25
    assert total_prompt_details.model_dump()["cache_write_tokens"] == 28
    assert (total_prompt_details.model_extra or {})["vendor_write_tokens"] == 9
    assert response_completion_details.accepted_prediction_tokens == 10
    assert response_completion_details.audio_tokens == 6
    assert response_completion_details.reasoning_tokens == 10
    assert response_completion_details.rejected_prediction_tokens == 13
    assert response_prompt_details.audio_tokens == 16
    assert response_prompt_details.cached_tokens == 25
    assert response_prompt_details.model_dump()["cache_write_tokens"] == 28
    assert (response_prompt_details.model_extra or {})["vendor_write_tokens"] == 9
    assert response_completion_details is not total_completion_details
    assert response_prompt_details is not total_prompt_details

    total_completion_details.audio_tokens = 99
    total_prompt_details.cached_tokens = 99
    assert response_completion_details.audio_tokens == 6
    assert response_prompt_details.cached_tokens == 25


def test_openai_usage_accumulates_sdk_extra_numeric_fields() -> None:
    total_usage = CompletionUsage.model_validate(
        {
            "completion_tokens": 1,
            "prompt_tokens": 2,
            "total_tokens": 3,
            "future_tokens": 4,
            "future_cost": 1.5,
        }
    )
    response_usage = CompletionUsage.model_validate(
        {
            "completion_tokens": 5,
            "prompt_tokens": 6,
            "total_tokens": 11,
            "future_tokens": 7,
            "future_cost": 2.25,
        }
    )
    response = _completion(1)
    response.usage = response_usage

    update_total_usage(response, total_usage)

    assert total_usage.model_extra is not None
    assert response_usage.model_extra is not None
    assert total_usage.model_extra["future_tokens"] == 11
    assert total_usage.model_extra["future_cost"] == 3.75
    assert response_usage.model_extra["future_tokens"] == 11
    assert response_usage.model_extra["future_cost"] == 3.75


def test_usage_carries_existing_non_numeric_fields_to_response() -> None:
    total_usage = CompletionUsage.model_validate(
        {
            "completion_tokens": 1,
            "prompt_tokens": 2,
            "total_tokens": 3,
            "service_tier": "scale",
        }
    )
    response_usage = CompletionUsage.model_validate(
        {
            "completion_tokens": 5,
            "prompt_tokens": 6,
            "total_tokens": 11,
        }
    )
    response = _completion(1)
    response.usage = response_usage

    update_total_usage(response, total_usage)

    assert response_usage.model_extra is not None
    assert response_usage.model_extra["service_tier"] == "scale"


def test_usage_response_non_numeric_fields_still_win() -> None:
    total_usage = CompletionUsage.model_validate(
        {
            "completion_tokens": 1,
            "prompt_tokens": 2,
            "total_tokens": 3,
            "service_tier": "scale",
        }
    )
    response_usage = CompletionUsage.model_validate(
        {
            "completion_tokens": 5,
            "prompt_tokens": 6,
            "total_tokens": 11,
            "service_tier": "default",
        }
    )
    response = _completion(1)
    response.usage = response_usage

    update_total_usage(response, total_usage)

    assert total_usage.model_extra is not None
    assert total_usage.model_extra["service_tier"] == "default"
    assert response_usage.model_extra is not None
    assert response_usage.model_extra["service_tier"] == "default"


def test_usage_accumulates_recursive_models_across_sdk_shape_changes() -> None:
    class Counters(BaseModel):
        billed: int
        label: str | None = None

    class Details(BaseModel):
        counters: Counters
        region: str

    class Usage(BaseModel):
        total: int
        details: Details | str | None = None

    total = Usage(total=0, details="legacy-shape")
    first = Usage(
        total=2,
        details=Details(counters=Counters(billed=4, label="new"), region="us"),
    )

    _accumulate_models(first, total)

    assert isinstance(total.details, Details)
    assert total.details.counters.billed == 4
    assert total.details.counters.label == "new"
    assert total.details.region == "us"

    second = Usage(total=3)
    _accumulate_models(second, total)

    assert second.total == 5
    assert second.details == total.details
    assert second.details is not total.details


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


def test_sync_patch_ignores_a_missing_optional_cache_backend() -> None:
    calls = 0

    def create(**_kwargs: Any) -> ChatCompletion:
        nonlocal calls
        calls += 1
        return _completion(11)

    cache = RecordingCache(fail_on_store=True)
    patched = patch_v2(create, Provider.OPENAI, Mode.JSON, default_model="default")

    result = patched(
        response_model=Answer,
        messages=[{"role": "user", "content": "eleven"}],
        cache=cache,
        cache_ttl="not-an-integer",
    )

    assert isinstance(result, Answer)
    assert result.value == 11
    assert calls == 1
    assert cache.ttls == [None]


def test_sync_patch_ignores_an_unsupported_cache() -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs)
        return _completion(12)

    patched = patch_v2(create, Provider.OPENAI, Mode.JSON, default_model="default")
    result = patched(
        response_model=Answer,
        messages=[{"role": "user", "content": "twelve"}],
        cache=UnsupportedCache(),
        cache_ttl=30,
    )

    assert isinstance(result, Answer)
    assert result.value == 12
    assert len(calls) == 1
    assert calls[0]["model"] == "default"
    assert "cache" not in calls[0]
    assert "cache_ttl" not in calls[0]


@pytest.mark.asyncio
async def test_async_patch_stores_a_cache_miss_and_returns_a_cache_hit() -> None:
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs)
        return _completion(13)

    cache = RecordingCache()
    patched = patch_v2(create, Provider.OPENAI, Mode.JSON, default_model="default")
    request = {"role": "user", "content": "thirteen"}

    first = await patched(
        response_model=Answer,
        messages=[dict(request)],
        cache=cache,
        cache_ttl=45,
    )
    second = await patched(
        response_model=Answer,
        messages=[dict(request)],
        cache=cache,
        cache_ttl=45,
    )

    assert isinstance(first, Answer)
    assert isinstance(second, Answer)
    assert first.value == 13
    assert second.value == 13
    assert len(calls) == 1
    assert calls[0]["model"] == "default"
    assert cache.ttls == [45]
    assert len(cache.values) == 1


@pytest.mark.asyncio
async def test_async_patch_ignores_a_missing_optional_cache_backend() -> None:
    calls = 0

    async def create(**_kwargs: Any) -> ChatCompletion:
        nonlocal calls
        calls += 1
        return _completion(17)

    cache = RecordingCache(fail_on_store=True)
    patched = patch_v2(create, Provider.OPENAI, Mode.JSON)

    result = await patched(
        response_model=Answer,
        model="test-model",
        messages=[{"role": "user", "content": "seventeen"}],
        cache=cache,
        cache_ttl=10,
    )

    assert isinstance(result, Answer)
    assert result.value == 17
    assert calls == 1
    assert cache.ttls == [10]


@pytest.mark.asyncio
async def test_async_patch_ignores_an_unsupported_cache() -> None:
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs)
        return _completion(18)

    patched = patch_v2(create, Provider.OPENAI, Mode.JSON, default_model="default")
    result = await patched(
        response_model=Answer,
        messages=[{"role": "user", "content": "eighteen"}],
        cache=UnsupportedCache(),
        cache_ttl=30,
    )

    assert isinstance(result, Answer)
    assert result.value == 18
    assert len(calls) == 1
    assert calls[0]["model"] == "default"
    assert "cache" not in calls[0]
    assert "cache_ttl" not in calls[0]


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


def test_retry_sync_wrapper_forwards_args_timeout_stream_and_non_strict_mode() -> None:
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def create(*args: Any, **kwargs: Any) -> ChatCompletion:
        seen.append((args, kwargs))
        return _completion("23")

    result = retry_sync(
        func=create,
        response_model=Answer,
        args=("request-id",),
        kwargs={
            "messages": [{"role": "user", "content": "twenty-three"}],
            "timeout": 30,
            "stream": True,
        },
        strict=False,
        mode=Mode.JSON,
        provider=Provider.OPENAI,
    )

    assert result == Answer(value=23)
    assert seen == [
        (
            ("request-id",),
            {
                "messages": [{"role": "user", "content": "twenty-three"}],
                "timeout": 30,
                "stream": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_retry_async_wrapper_reasks_emits_parse_hook_and_forwards_args() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    parse_errors: list[tuple[Exception, dict[str, Any]]] = []
    responses = iter([_completion("invalid"), _completion("29")])

    async def create(*args: Any, **kwargs: Any) -> ChatCompletion:
        calls.append((args, kwargs))
        return next(responses)

    hooks = Hooks()
    hooks.on(
        "parse:error",
        lambda error, **metadata: parse_errors.append((error, metadata)),
    )

    result = await retry_async(
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
    )

    assert result == Answer(value=29)
    assert len(calls) == 2
    assert calls[0][0] == ("request-id",)
    assert calls[0][1]["timeout"] == 30.0
    assert calls[0][1]["stream"] is True
    assert calls[1][1]["messages"][-1]["role"] == "user"
    assert "Correct your JSON ONLY RESPONSE" in calls[1][1]["messages"][-1]["content"]
    assert len(parse_errors) == 1
    assert parse_errors[0][1] == {
        "attempt_number": 1,
        "max_attempts": 2,
        "is_last_attempt": False,
    }


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


def test_retry_sync_wrapper_defaults_are_strict_and_allow_one_retry() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
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

    def create(*args: Any, **kwargs: Any) -> ChatCompletion:
        calls.append((args, kwargs))
        return _completion("41")

    hooks = Hooks()
    hooks.on(
        "completion:kwargs",
        lambda **kwargs: completion_kwargs.append(deepcopy(kwargs)),
    )
    hooks.on("completion:response", lambda response: responses.append(response))
    hooks.on(
        "parse:error",
        lambda error, **metadata: parse_errors.append((error, metadata)),
    )
    hooks.on(
        "completion:last_attempt",
        lambda error, **metadata: last_attempts.append((error, metadata)),
    )
    context = {"tenant": "sync"}

    with pytest.raises(InstructorRetryException) as exc_info:
        retry_sync(
            func=create,
            response_model=ContextAnswer,
            args=("sync-request",),
            kwargs={"messages": [{"role": "user", "content": "strict value"}]},
            context=context,
            mode=Mode.JSON,
            provider=Provider.OPENAI,
            hooks=hooks,
        )

    assert len(calls) == 2
    assert all(args == ("sync-request",) for args, _ in calls)
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
    assert len(parse_errors) == 2
    assert all(isinstance(error, ValidationError) for error, _ in parse_errors)
    assert all("valid integer" in str(error) for error, _ in parse_errors)
    assert [metadata for _, metadata in parse_errors] == [
        {"attempt_number": 1, "max_attempts": 2, "is_last_attempt": False},
        {"attempt_number": 2, "max_attempts": 2, "is_last_attempt": True},
    ]
    assert len(last_attempts) == 1
    assert isinstance(last_attempts[0][0], ValidationError)
    assert last_attempts[0][1] == {
        "attempt_number": 2,
        "max_attempts": 2,
        "is_last_attempt": True,
    }
    assert exc_info.value.n_attempts == 2
    assert len(exc_info.value.failed_attempts or []) == 2


@pytest.mark.asyncio
async def test_retry_async_wrapper_defaults_are_strict_and_allow_one_retry() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
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

    async def create(*args: Any, **kwargs: Any) -> ChatCompletion:
        calls.append((args, kwargs))
        return _completion("43")

    hooks = Hooks()
    hooks.on(
        "completion:kwargs",
        lambda **kwargs: completion_kwargs.append(deepcopy(kwargs)),
    )
    hooks.on("completion:response", lambda response: responses.append(response))
    hooks.on(
        "parse:error",
        lambda error, **metadata: parse_errors.append((error, metadata)),
    )
    hooks.on(
        "completion:last_attempt",
        lambda error, **metadata: last_attempts.append((error, metadata)),
    )
    context = {"tenant": "async"}

    with pytest.raises(InstructorRetryException) as exc_info:
        await retry_async(
            func=create,
            response_model=ContextAnswer,
            args=("async-request",),
            kwargs={"messages": [{"role": "user", "content": "strict value"}]},
            context=context,
            mode=Mode.JSON,
            provider=Provider.OPENAI,
            hooks=hooks,
        )

    assert len(calls) == 2
    assert all(args == ("async-request",) for args, _ in calls)
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
    assert len(parse_errors) == 2
    assert all(isinstance(error, ValidationError) for error, _ in parse_errors)
    assert all("valid integer" in str(error) for error, _ in parse_errors)
    assert [metadata for _, metadata in parse_errors] == [
        {"attempt_number": 1, "max_attempts": 2, "is_last_attempt": False},
        {"attempt_number": 2, "max_attempts": 2, "is_last_attempt": True},
    ]
    assert len(last_attempts) == 1
    assert isinstance(last_attempts[0][0], ValidationError)
    assert last_attempts[0][1] == {
        "attempt_number": 2,
        "max_attempts": 2,
        "is_last_attempt": True,
    }
    assert exc_info.value.n_attempts == 2
    assert len(exc_info.value.failed_attempts or []) == 2


def test_retry_sync_does_not_retry_an_incomplete_streaming_response() -> None:
    response = _completion("partial", finish_reason="length")
    calls = 0

    def create(**_kwargs: Any) -> ChatCompletion:
        nonlocal calls
        calls += 1
        return response

    with pytest.raises(IncompleteOutputException) as exc_info:
        retry_sync_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=3,
            args=(),
            kwargs={"messages": [], "stream": True},
            strict=True,
        )

    assert exc_info.value.last_completion is response
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_async_does_not_retry_an_incomplete_streaming_response() -> None:
    response = _completion("partial", finish_reason="length")
    calls = 0

    async def create(**_kwargs: Any) -> ChatCompletion:
        nonlocal calls
        calls += 1
        return response

    with pytest.raises(IncompleteOutputException) as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=3,
            args=(),
            kwargs={"messages": [], "stream": True},
            strict=True,
        )

    assert exc_info.value.last_completion is response
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_async_wraps_an_api_error_without_hooks_or_retries() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def create(*args: Any, **kwargs: Any) -> ChatCompletion:
        calls.append((args, kwargs))
        raise ConnectionError("provider is unavailable")

    messages = [{"role": "user", "content": "answer once"}]
    with pytest.raises(
        InstructorRetryException, match="provider is unavailable"
    ) as exc:
        await retry_async_v2(
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


def test_retry_sync_reports_a_policy_that_yields_no_attempts() -> None:
    calls = 0

    def create(**_kwargs: Any) -> ChatCompletion:
        nonlocal calls
        calls += 1
        return _completion(31)

    with pytest.raises(InstructorRetryException, match="Unknown error") as exc_info:
        retry_sync_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=NoAttemptsRetrying(),
            args=(),
            kwargs={"messages": [{"role": "user", "content": "unused"}]},
            strict=True,
        )

    assert exc_info.value.n_attempts == 0
    assert exc_info.value.create_kwargs == {
        "messages": [{"role": "user", "content": "unused"}]
    }
    assert calls == 0


@pytest.mark.asyncio
async def test_retry_async_reports_a_policy_that_yields_no_attempts() -> None:
    calls = 0

    async def create(**_kwargs: Any) -> ChatCompletion:
        nonlocal calls
        calls += 1
        return _completion(37)

    with pytest.raises(InstructorRetryException, match="Unknown error") as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=NoAttemptsAsyncRetrying(),
            args=(),
            kwargs={"messages": [{"role": "user", "content": "unused"}]},
            strict=True,
        )

    assert exc_info.value.n_attempts == 0
    assert exc_info.value.create_kwargs == {
        "messages": [{"role": "user", "content": "unused"}]
    }
    assert calls == 0
