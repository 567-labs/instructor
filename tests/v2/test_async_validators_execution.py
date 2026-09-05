"""Regression tests for GitHub issue #2528: @async_field_validator and
@async_model_validator were decorator-only -- nothing in the retry path ever
awaited them, so declared async validators silently never ran.

These drive the real v2 retry functions (`retry_async_v2` / `retry_sync_v2`)
against fake completions, the same pattern used in test_messages_not_mutated.py,
so the fix is verified end-to-end rather than by calling the validator helpers
directly.
"""

from __future__ import annotations

from typing import Any

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from instructor.v2.core.errors import (
    AsyncValidationError,
    ConfigurationError,
    InstructorRetryException,
)
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.core.retry import retry_async_v2, retry_sync_v2
from instructor.v2.validation import async_field_validator, async_model_validator


def _tool_call_response(arguments: str, call_id: str = "call_1") -> ChatCompletion:
    tool_call = ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name="Answer", arguments=arguments),
    )
    message = ChatCompletionMessage(
        role="assistant", content=None, tool_calls=[tool_call]
    )
    choice = Choice(index=0, message=message, finish_reason="tool_calls", logprobs=None)
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[choice],
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=5, prompt_tokens=10, total_tokens=15),
    )


def _build_kwargs(response_model: type[BaseModel]) -> tuple[Any, dict[str, Any]]:
    handlers = mode_registry.get_handlers(Provider.OPENAI, Mode.TOOLS)
    return handlers.request_handler(
        response_model=response_model,
        kwargs={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "go"}],
        },
    )


class Email(BaseModel):
    address: str

    @async_field_validator("address")
    async def must_contain_at(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError(f"{value!r} is not a valid email address")
        return value.lower()


@pytest.mark.asyncio
async def test_async_field_validator_transforms_value_on_success():
    calls = {"n": 0}

    async def fake_create(*_a: Any, **_k: Any) -> ChatCompletion:
        calls["n"] += 1
        return _tool_call_response('{"address": "Jane@Example.com"}')

    response_model, kwargs = _build_kwargs(Email)
    result = await retry_async_v2(
        func=fake_create,
        response_model=response_model,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=2,
        args=(),
        kwargs=kwargs,
        strict=True,
        hooks=None,
    )

    assert calls["n"] == 1
    assert isinstance(result, Email)
    assert result.address == "jane@example.com"


@pytest.mark.asyncio
async def test_async_field_validator_failure_triggers_reask_then_succeeds():
    calls = {"n": 0}

    async def fake_create(*_a: Any, **_k: Any) -> ChatCompletion:
        calls["n"] += 1
        if calls["n"] == 1:
            return _tool_call_response('{"address": "not-an-email"}', call_id="call_1")
        return _tool_call_response('{"address": "jane@example.com"}', call_id="call_2")

    response_model, kwargs = _build_kwargs(Email)
    result = await retry_async_v2(
        func=fake_create,
        response_model=response_model,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=2,
        args=(),
        kwargs=kwargs,
        strict=True,
        hooks=None,
    )

    assert calls["n"] == 2, "validator failure must trigger exactly one reask"
    assert result.address == "jane@example.com"


@pytest.mark.asyncio
async def test_async_field_validator_exhausting_retries_raises_instructor_retry_exception():
    calls = {"n": 0}

    async def fake_create(*_a: Any, **_k: Any) -> ChatCompletion:
        calls["n"] += 1
        return _tool_call_response(
            '{"address": "not-an-email"}', call_id=f"call_{calls['n']}"
        )

    response_model, kwargs = _build_kwargs(Email)
    with pytest.raises(InstructorRetryException) as exc_info:
        await retry_async_v2(
            func=fake_create,
            response_model=response_model,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=1,
            args=(),
            kwargs=kwargs,
            strict=True,
            hooks=None,
        )

    assert calls["n"] == 2  # initial attempt + 1 retry
    assert isinstance(
        exc_info.value.failed_attempts[-1].exception, AsyncValidationError
    )


class Account(BaseModel):
    email: Email

    @async_model_validator()
    async def normalize(self) -> Account:
        return self.model_copy(update={"email": self.email})


@pytest.mark.asyncio
async def test_async_validators_recurse_into_nested_models():
    """A nested BaseModel field's async validator must run too, and its
    failure must surface through the parent's validation error."""
    calls = {"n": 0}

    async def fake_create(*_a: Any, **_k: Any) -> ChatCompletion:
        calls["n"] += 1
        return _tool_call_response('{"email": {"address": "still-invalid"}}')

    response_model, kwargs = _build_kwargs(Account)
    with pytest.raises(InstructorRetryException) as exc_info:
        await retry_async_v2(
            func=fake_create,
            response_model=response_model,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=0,
            args=(),
            kwargs=kwargs,
            strict=True,
            hooks=None,
        )

    assert exc_info.value.failed_attempts
    error = exc_info.value.failed_attempts[-1].exception
    assert isinstance(error, AsyncValidationError)
    assert "still-invalid" in str(error)


def test_sync_client_fails_fast_for_response_model_with_async_validators():
    """A sync client can never await an async validator; it must raise a
    clear ConfigurationError instead of silently skipping validation."""
    calls = {"n": 0}

    def fake_create(*_a: Any, **_k: Any) -> ChatCompletion:
        calls["n"] += 1
        return _tool_call_response('{"address": "not-an-email"}')

    response_model, kwargs = _build_kwargs(Email)
    with pytest.raises(ConfigurationError, match="async client"):
        retry_sync_v2(
            func=fake_create,
            response_model=response_model,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=2,
            args=(),
            kwargs=kwargs,
            strict=True,
            hooks=None,
        )

    assert calls["n"] == 0, "must fail before ever calling the API"


def test_sync_client_unaffected_when_model_has_no_async_validators():
    class Plain(BaseModel):
        name: str

    def fake_create(*_a: Any, **_k: Any) -> ChatCompletion:
        return _tool_call_response('{"name": "Ada"}')

    response_model, kwargs = _build_kwargs(Plain)
    result = retry_sync_v2(
        func=fake_create,
        response_model=response_model,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=2,
        args=(),
        kwargs=kwargs,
        strict=True,
        hooks=None,
    )

    assert result.name == "Ada"
