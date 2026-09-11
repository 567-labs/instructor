"""Regression tests for the messages= aliasing bug (see GitHub issue #2417).

`client.create()` must treat `messages=` as a read-only input. Internally,
`prepare_request` must never hand back the caller's own list object, or
leave the caller's message dicts open to in-place mutation, because both
`prepare_request` itself (JSON/MD_JSON's system-message injection) and
`handle_reask` (the retry path) mutate whatever `messages` list/dicts they
are handed.

Note on the aliasing check: comparing `new_kwargs["messages"] is not
caller_messages` alone is not sufficient -- `OpenAIJSONHandler` and
`OpenAIMDJSONHandler` mutate the *input* list/dicts in place before
rebuilding `messages` into a genuinely new list via
`merge_consecutive_messages()`, so an identity check on the final result
would pass even though the caller's original objects were already
corrupted. These tests instead snapshot `caller_messages` with `deepcopy`
before the call and assert it is unchanged afterward.

The provider lists below are imported directly from the shared OpenAI-compat
handler module so this test stays in sync with the source registrations
rather than duplicating them.

Independently-implemented handlers (Mistral, Cohere, Writer, xAI, OpenRouter,
Requesty) reproduce the same anti-pattern in their own provider-specific code
and were previously tracked here as known-broken `xfail` cases. They are now
fixed too and folded into `FIXED_PAIRS` below.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
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

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.core.retry import retry_sync_v2
from instructor.v2.providers.openai.handlers import (
    OPENAI_COMPAT_PROVIDERS,
    OPENAI_JSON_SCHEMA_PROVIDERS,
    OPENAI_PARALLEL_TOOL_PROVIDERS,
)


class Answer(BaseModel):
    name: str
    age: int


# (provider, mode) pairs whose handlers were patched to stop aliasing
# `messages`. Covers both the shared OpenAI-compat handler classes and the
# independently-implemented provider-specific handlers (Mistral, Cohere,
# Writer, xAI, OpenRouter, Requesty) that reproduced the same anti-pattern
# separately.
FIXED_PAIRS: list[tuple[Provider, Mode]] = [
    *((provider, Mode.TOOLS) for provider in OPENAI_COMPAT_PROVIDERS),
    *((provider, Mode.JSON_SCHEMA) for provider in OPENAI_JSON_SCHEMA_PROVIDERS),
    *((provider, Mode.PARALLEL_TOOLS) for provider in OPENAI_PARALLEL_TOOL_PROVIDERS),
    *((provider, Mode.JSON) for provider in OPENAI_COMPAT_PROVIDERS),
    *((provider, Mode.MD_JSON) for provider in OPENAI_COMPAT_PROVIDERS),
    (Provider.OPENAI, Mode.RESPONSES_TOOLS),
    (Provider.MISTRAL, Mode.TOOLS),
    (Provider.MISTRAL, Mode.JSON_SCHEMA),
    (Provider.MISTRAL, Mode.MD_JSON),
    (Provider.COHERE, Mode.JSON_SCHEMA),
    (Provider.COHERE, Mode.MD_JSON),
    (Provider.WRITER, Mode.TOOLS),
    (Provider.WRITER, Mode.JSON_SCHEMA),
    (Provider.WRITER, Mode.MD_JSON),
    (Provider.XAI, Mode.TOOLS),
    (Provider.XAI, Mode.PARALLEL_TOOLS),
    (Provider.XAI, Mode.JSON_SCHEMA),
    (Provider.XAI, Mode.MD_JSON),
    (Provider.OPENROUTER, Mode.JSON_SCHEMA),
    (Provider.REQUESTY, Mode.JSON_SCHEMA),
]


@pytest.mark.parametrize(
    ("provider", "mode"),
    FIXED_PAIRS,
    ids=[f"{provider.value}-{mode.value}" for provider, mode in FIXED_PAIRS],
)
def test_prepare_request_does_not_alias_caller_messages(
    provider: Provider, mode: Mode
) -> None:
    caller_messages = [{"role": "user", "content": "hi"}]
    original_snapshot = deepcopy(caller_messages)
    kwargs: dict[str, Any] = {"model": "test", "messages": caller_messages}
    response_model: Any = Iterable[Answer] if mode is Mode.PARALLEL_TOOLS else Answer

    handlers = mode_registry.get_handlers(provider, mode)
    _, new_kwargs = handlers.request_handler(
        response_model=response_model, kwargs=kwargs
    )

    assert new_kwargs["messages"] is not caller_messages
    # Deep-content check, not just identity: a handler can rebuild
    # `new_kwargs["messages"]` into a fresh list while still having mutated
    # the caller's original list/dicts in place before doing so.
    assert caller_messages == original_snapshot


def _make_tool_call_response(arguments: str, call_id: str) -> ChatCompletion:
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


def test_client_create_does_not_mutate_caller_messages_after_reask() -> None:
    """End-to-end: a validation failure followed by a successful retry must
    leave the caller's original `messages` list completely untouched."""
    call_count = {"n": 0}

    def fake_openai_create(*_args: Any, **_kwargs: Any) -> ChatCompletion:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _make_tool_call_response('{"name": "Ada"}', "call_1")
        return _make_tool_call_response('{"name": "Ada", "age": 37}', "call_2")

    caller_messages = [{"role": "user", "content": "Ada is 37 years old"}]

    handlers = mode_registry.get_handlers(Provider.OPENAI, Mode.TOOLS)
    response_model, new_kwargs = handlers.request_handler(
        response_model=Answer,
        kwargs={"model": "gpt-4o-mini", "messages": caller_messages},
    )

    result = retry_sync_v2(
        func=fake_openai_create,
        response_model=response_model,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=2,
        args=(),
        kwargs=new_kwargs,
        strict=True,
        hooks=None,
    )

    assert isinstance(result, Answer)
    assert result.name == "Ada"
    assert result.age == 37
    assert caller_messages == [{"role": "user", "content": "Ada is 37 years old"}]


REASK_MUTATION_PAIRS: list[tuple[Provider, Mode]] = [
    (Provider.MISTRAL, Mode.TOOLS),
    (Provider.MISTRAL, Mode.JSON_SCHEMA),
    (Provider.COHERE, Mode.JSON_SCHEMA),
    (Provider.WRITER, Mode.TOOLS),
    (Provider.WRITER, Mode.JSON_SCHEMA),
    (Provider.XAI, Mode.TOOLS),
    (Provider.XAI, Mode.PARALLEL_TOOLS),
    (Provider.XAI, Mode.JSON_SCHEMA),
    (Provider.OPENROUTER, Mode.JSON_SCHEMA),
    (Provider.REQUESTY, Mode.JSON_SCHEMA),
]


@pytest.mark.parametrize(
    ("provider", "mode"),
    REASK_MUTATION_PAIRS,
    ids=[f"{provider.value}-{mode.value}" for provider, mode in REASK_MUTATION_PAIRS],
)
def test_reask_does_not_mutate_caller_messages(provider: Provider, mode: Mode) -> None:
    """Simulates the actual retry path (prepare_request -> handle_reask) for
    provider-specific handlers, since `prepare_request` alone never touches
    `messages` for these modes -- the append only happens once a reask
    actually runs, which a prepare_request-only check would miss."""
    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function=Function(name="Answer", arguments='{"name": "Ada"}'),
    )
    response = ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content="stub", tool_calls=[tool_call]
                ),
                finish_reason="tool_calls",
                logprobs=None,
            )
        ],
        created=0,
        model="m",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
    )
    exception = ValueError("1 validation error for Answer\nage\n  Field required")

    caller_messages = [{"role": "user", "content": "hi"}]
    original_snapshot = deepcopy(caller_messages)
    response_model: Any = Iterable[Answer] if mode is Mode.PARALLEL_TOOLS else Answer

    handlers = mode_registry.get_handlers(provider, mode)
    _, new_kwargs = handlers.request_handler(
        response_model=response_model,
        kwargs={"model": "test", "messages": caller_messages},
    )
    handlers.reask_handler(new_kwargs, response, exception)

    assert caller_messages == original_snapshot
