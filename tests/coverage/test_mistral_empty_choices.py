"""Mistral handlers must survive responses without a usable completion choice.

Mistral (and gateways in front of it) can return HTTP 200 with an empty
``choices`` list, for example when a prompt is rejected by content filtering.
The parse layer must turn that into a retryable ``ResponseParsingError`` --
not crash with ``IndexError`` -- and the reask handlers must append the
validation feedback instead of indexing ``response.choices[0]``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import InstructorRetryException, ResponseParsingError
from instructor.v2.core.messages import isolate_retry_kwargs
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.core.retry import retry_sync_v2

# Importing the module registers the lazy handlers.
import instructor.v2.providers.mistral.handlers  # noqa: E402, F401


class Answer(BaseModel):
    name: str


def _empty_choices_response() -> Any:
    return SimpleNamespace(choices=[])


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


CASES: list[tuple[Provider, Mode]] = [
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
    kwargs = isolate_retry_kwargs(
        {"model": "mistral-large-latest", "messages": caller_messages}
    )

    reasked = handlers.reask_handler(
        kwargs=kwargs,
        response=_empty_choices_response(),
        exception=ValueError("model returned no choice"),
    )

    # The caller's own conversation is untouched, and with no assistant message
    # to replay the reask only carries the correction prompt.
    assert caller_messages == [{"role": "user", "content": "Ada is 37 years old"}]
    assert [message["role"] for message in reasked["messages"]] == ["user", "user"]


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


def test_empty_choices_response_retries_instead_of_crashing() -> None:
    attempts: list[int] = []

    def fake_create(**_kwargs: Any) -> Any:
        attempts.append(1)
        return _empty_choices_response()

    with pytest.raises(InstructorRetryException) as excinfo:
        retry_sync_v2(
            func=fake_create,
            response_model=Answer,
            provider=Provider.MISTRAL,
            mode=Mode.TOOLS,
            context=None,
            max_retries=1,
            args=(),
            kwargs={
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": "hi"}],
            },
            strict=True,
        )

    assert len(attempts) == 2
    assert excinfo.value.n_attempts == 2
    assert isinstance(excinfo.value.__cause__, ResponseParsingError)
