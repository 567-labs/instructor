"""Reask handlers must survive responses without a usable completion choice.

OpenAI-compatible gateways and Azure OpenAI can return HTTP 200 with an empty
``choices`` list (for example when a prompt is rejected by content
filtering). The parse layer already turns that into a retryable
``ResponseParsingError``; the retry loop then hands the same response to the
reask handler, which must append the validation feedback instead of indexing
``response.choices[0]`` and crashing with ``IndexError``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from instructor.v2.core.errors import InstructorRetryException, ResponseParsingError
from instructor.v2.core.messages import isolate_retry_kwargs
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.core.retry import retry_sync_v2

# Importing the modules registers the lazy handlers.
import instructor.v2.providers.openai.handlers  # noqa: E402, F401
import instructor.v2.providers.perplexity.handlers  # noqa: E402, F401


class Answer(BaseModel):
    name: str


def _empty_choices_response(provider: Provider) -> ChatCompletion | SimpleNamespace:
    if provider is Provider.MISTRAL:
        return SimpleNamespace(choices=[])
    return ChatCompletion(
        id="chatcmpl-empty-choices",
        choices=[],
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=None,
    )


CASES: list[tuple[Provider, Mode]] = [
    (Provider.OPENAI, Mode.TOOLS),
    (Provider.OPENAI, Mode.JSON),
    (Provider.OPENAI, Mode.MD_JSON),
    (Provider.PERPLEXITY, Mode.MD_JSON),
    (Provider.MISTRAL, Mode.TOOLS),
    (Provider.MISTRAL, Mode.JSON_SCHEMA),
    (Provider.MISTRAL, Mode.MD_JSON),
]


@pytest.mark.parametrize(
    ("provider", "mode"), CASES, ids=[f"{p.value}-{m.value}" for p, m in CASES]
)
def test_reask_without_choices_appends_only_the_correction(
    provider: Provider, mode: Mode
) -> None:
    handlers = mode_registry.get_handlers(provider, mode)
    caller_messages = [{"role": "user", "content": "Ada is 37 years old"}]
    kwargs = isolate_retry_kwargs({"model": "gpt-4o-mini", "messages": caller_messages})

    reasked = handlers.reask_handler(
        kwargs=kwargs,
        response=_empty_choices_response(provider),
        exception=ValueError("model returned no choice"),
    )

    # The caller's own conversation is untouched, and with no assistant message
    # to replay the reask only carries the correction prompt.
    assert caller_messages == [{"role": "user", "content": "Ada is 37 years old"}]
    assert [message["role"] for message in reasked["messages"]] == ["user", "user"]


@pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.MISTRAL])
def test_empty_choices_response_retries_instead_of_crashing(provider: Provider) -> None:
    calls = 0

    def fake_create(**_kwargs: Any) -> ChatCompletion | SimpleNamespace:
        nonlocal calls
        calls += 1
        return _empty_choices_response(provider)

    with pytest.raises(InstructorRetryException) as excinfo:
        retry_sync_v2(
            func=fake_create,
            response_model=Answer,
            provider=provider,
            mode=Mode.TOOLS,
            context=None,
            max_retries=1,
            args=(),
            kwargs={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
            strict=True,
        )

    # The empty-choices response is retried like any other parse failure, and the
    # final error is the usual retry exception rather than an IndexError from the
    # reask handler.
    assert isinstance(excinfo.value.__cause__, ResponseParsingError)
    assert calls == excinfo.value.n_attempts == 2
    assert excinfo.value.failed_attempts is not None
    assert len(excinfo.value.failed_attempts) == 2


def _tool_call_response(arguments: str = '{"name": "Ada"}') -> Any:
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="Answer", arguments=arguments),
    )
    message = SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
        model_dump=lambda: {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Answer", "arguments": arguments},
                }
            ],
        },
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_reask_with_choice_still_replays_assistant_message() -> None:
    """Positive control: a usable choice keeps the old replay behavior."""
    handlers = mode_registry.get_handlers(Provider.MISTRAL, Mode.TOOLS)
    kwargs = isolate_retry_kwargs(
        {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    reasked = handlers.reask_handler(
        kwargs=kwargs,
        response=_tool_call_response('{"wrong": true}'),
        exception=ValueError("boom"),
    )

    assert [message["role"] for message in reasked["messages"]] == [
        "user",
        "assistant",
        "tool",
    ]
