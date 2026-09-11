"""Supported usage contracts, including the Anthropic thinking-token regression.

These positive assertions should survive future accounting fixes. Known gaps
live separately in test_usage_accounting_audit.py. Only repeated SDK envelope
fields are shared; provider usage payloads and expected arithmetic stay local.
"""

from __future__ import annotations

from typing import Any

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from instructor.v2.core.providers import Provider
from instructor.v2.core.retry import (
    _initialize_usage,
    _usage_snapshot,
    _usage_total_tokens,
)
from instructor.v2.core.usage import update_total_usage


def anthropic_message(usage: dict[str, Any]) -> Any:
    types = pytest.importorskip("anthropic.types")
    return types.Message.model_validate(
        {
            "id": "audit",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "audit",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }
    )


def test_anthropic_thinking_detail_accumulates_across_sdk_shapes() -> None:
    pytest.importorskip("anthropic.types")
    total = _initialize_usage(Provider.ANTHROPIC)
    for _ in range(2):
        message = anthropic_message(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "output_tokens_details": {"thinking_tokens": 12},
            }
        )
        update_total_usage(message, total)
    assert total.output_tokens == 40
    assert total.model_dump()["output_tokens_details"]["thinking_tokens"] == 24
    assert message.usage.model_dump()["output_tokens_details"]["thinking_tokens"] == 24


def test_openai_subsets_are_not_added_again_to_total() -> None:
    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 80, "audio_tokens": 10},
            "completion_tokens_details": {
                "reasoning_tokens": 30,
                "accepted_prediction_tokens": 5,
                "rejected_prediction_tokens": 3,
            },
            "is_billable": True,
        }
    )
    total = _initialize_usage(Provider.OPENAI)
    for _ in range(2):
        update_total_usage(
            ChatCompletion(
                id="audit",
                created=0,
                model="audit",
                object="chat.completion",
                choices=[],
                usage=usage.model_copy(deep=True),
            ),
            total,
        )
    assert _usage_total_tokens(total) == 300
    assert total.prompt_tokens_details.cached_tokens == 160
    assert total.completion_tokens_details.reasoning_tokens == 60
    assert total.model_extra["is_billable"] is True


def test_anthropic_cache_ttl_and_tools_have_separate_semantics() -> None:
    pytest.importorskip("anthropic.types")
    total = _initialize_usage(Provider.ANTHROPIC)
    for _ in range(2):
        message = anthropic_message(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 30,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 20,
                    "ephemeral_1h_input_tokens": 10,
                },
                "server_tool_use": {
                    "web_search_requests": 2,
                    "web_fetch_requests": 1,
                },
            }
        )
        update_total_usage(message, total)
    assert _usage_total_tokens(total) == 760
    assert total.cache_creation.ephemeral_1h_input_tokens == 20
    assert total.server_tool_use.web_search_requests == 4


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [(0, 0, 0), (None, 6, 6), (12, None, None), (True, 6, 6), (12, False, False)],
)
def test_anthropic_dictionary_thinking_preserves_non_counts(
    previous: int | None, current: int | None, expected: int | None
) -> None:
    types = pytest.importorskip("anthropic.types")
    if "output_tokens_details" in types.Usage.model_fields:
        # Invalid extra-dictionary payloads apply only before this field is typed.
        pytest.skip()
    from instructor.v2.providers.anthropic.usage import initialize_usage
    from instructor.v2.providers.anthropic.usage import update_total_usage as accumulate

    total = initialize_usage()
    for count, unknown in [(previous, 100), (current, 200)]:
        usage = types.Usage.model_validate(
            {
                "input_tokens": 1,
                "output_tokens": 20,
                "output_tokens_details": {
                    "thinking_tokens": count,
                    "unknown_metadata": unknown,
                },
            }
        )
        assert accumulate(usage, total)
    details = total.model_dump()["output_tokens_details"]
    assert details["thinking_tokens"] == expected
    assert type(details["thinking_tokens"]) is type(expected)
    assert details["unknown_metadata"] == 200


@pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.ANTHROPIC])
def test_supported_usage_keeps_sdk_identity_and_isolates_snapshots(
    provider: Provider,
) -> None:
    if provider is Provider.ANTHROPIC:
        response = anthropic_message({"input_tokens": 10, "output_tokens": 5})
    else:
        response = ChatCompletion(
            id="audit",
            created=0,
            model="audit",
            object="chat.completion",
            choices=[],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
    usage_type = type(response.usage)
    total = _initialize_usage(provider)
    assert update_total_usage(response, total) is response
    assert type(response.usage) is usage_type
    assert _usage_total_tokens(total) == 15
    snapshot = _usage_snapshot(total)
    assert snapshot.model_dump() == total.model_dump()
    update_total_usage(response.model_copy(deep=True), total)
    assert _usage_total_tokens(total) == 30
    assert _usage_total_tokens(snapshot) == 15
