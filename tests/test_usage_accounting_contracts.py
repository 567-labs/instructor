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
from openai.types.responses import Response

from instructor.v2.core.providers import Provider
from instructor.v2.core.retry import (
    _initialize_usage,
    _usage_snapshot,
    _usage_total_tokens,
)
from instructor.v2.core.usage import has_compatible_usage, update_total_usage


def _responses_response(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cached_tokens: int,
    reasoning_tokens: int,
) -> Response:
    """Build an OpenAI Responses-API ``Response`` the way the SDK returns one."""
    return Response.model_validate(
        {
            "id": "usage-audit",
            "created_at": 0,
            "object": "response",
            "model": "audit",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_tokens_details": {
                    "cached_tokens": cached_tokens,
                    "cache_write_tokens": 0,
                },
                "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            },
        }
    )


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


def test_openai_responses_api_usage_accumulates_into_chat_total() -> None:
    """OpenAI Responses API ``ResponseUsage`` must accumulate like Chat usage.

    The Responses API reports ``input_tokens``/``output_tokens`` while the
    running total is a Chat ``CompletionUsage`` (``prompt_tokens``/
    ``completion_tokens``). Without mapping, Responses-API retries silently
    drop every token, so ``token_budget`` enforcement and cumulative usage
    both read zero.
    """
    total = _initialize_usage(Provider.OPENAI)
    first = _responses_response(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cached_tokens=8,
        reasoning_tokens=2,
    )
    assert has_compatible_usage(first, total) is True

    update_total_usage(first, total)
    second = _responses_response(
        input_tokens=20,
        output_tokens=7,
        total_tokens=27,
        cached_tokens=5,
        reasoning_tokens=3,
    )
    update_total_usage(second, total)

    assert _usage_total_tokens(total) == 42
    assert total.prompt_tokens == 30
    assert total.completion_tokens == 12
    assert total.prompt_tokens_details.cached_tokens == 13
    assert total.completion_tokens_details.reasoning_tokens == 5
    # Latest response echoes the running totals back (matches Chat behavior).
    assert second.usage is not None
    assert second.usage.input_tokens_details is not None
    assert second.usage.output_tokens_details is not None
    assert second.usage.total_tokens == 42
    assert second.usage.input_tokens == 30
    assert second.usage.output_tokens == 12
    assert second.usage.input_tokens_details.cached_tokens == 13
    assert second.usage.output_tokens_details.reasoning_tokens == 5


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


def test_absent_responses_sdk_leaves_unknown_usage_unchanged(monkeypatch):
    import sys
    from types import SimpleNamespace
    from instructor.v2.core.usage import has_compatible_usage, update_total_usage

    monkeypatch.setitem(sys.modules, "openai.types.responses", None)
    total = CompletionUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3)
    response = SimpleNamespace(usage=SimpleNamespace(custom_tokens=5))
    assert not has_compatible_usage(response, total)
    assert update_total_usage(response, total) is response
    assert total.total_tokens == 3


def _beta_message(**extra: Any) -> Any:
    beta = pytest.importorskip("anthropic.types.beta")
    return beta.BetaMessage.model_validate(
        {
            "id": "beta-usage",
            "type": "message",
            "role": "assistant",
            "model": "audit",
            "content": [],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 3,
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_5m_input_tokens": 2,
                },
                "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 2},
                "output_tokens_details": {"thinking_tokens": 2},
                **extra,
            },
        }
    )


def test_native_beta_usage_accumulates_without_losing_serialized_details() -> None:
    from instructor.v2.core.usage import _prepare_usage_accumulator

    first = _beta_message(
        audit_metadata={"keep": "first"}, audit_number=7, total_tokens=999
    )
    second = _beta_message(
        audit_metadata={"keep": "last"}, audit_number=9, total_tokens=1000
    )
    original = _initialize_usage(Provider.ANTHROPIC)
    total = _prepare_usage_accumulator(first, original)
    assert type(total) is type(first.usage)
    assert original.input_tokens == 0
    assert has_compatible_usage(first, total)
    update_total_usage(first, total)
    snapshot = _usage_snapshot(total)
    update_total_usage(second, total)
    assert total.input_tokens == 20
    assert total.output_tokens == 10
    assert _usage_total_tokens(total) == 40
    dumped = total.model_dump()
    assert dumped["server_tool_use"] == {
        "web_search_requests": 2,
        "web_fetch_requests": 4,
    }
    assert dumped["cache_creation"]["ephemeral_5m_input_tokens"] == 4
    assert dumped["output_tokens_details"]["thinking_tokens"] == 4
    assert dumped["audit_metadata"] == {"keep": "last"}
    assert dumped["audit_number"] == 9
    assert dumped["total_tokens"] == 1000
    assert second.usage.model_dump() == dumped
    type(total).model_validate(dumped)
    second.usage.audit_metadata["keep"] = "changed"
    assert total.audit_metadata == {"keep": "last"}
    assert snapshot.input_tokens == 10
    assert snapshot.audit_metadata == {"keep": "first"}


@pytest.mark.parametrize("last", ["redeemed", "not_applied", None])
def test_beta_fallback_credit_keeps_latest_native_variant(last: str | None) -> None:
    from instructor.v2.core.usage import _prepare_usage_accumulator

    beta = pytest.importorskip("anthropic.types.beta")
    if "fallback_credit" not in beta.BetaUsage.model_fields:
        pytest.skip("SDK predates native fallback credit metadata")
    first_status = (
        {"type": "redeemed"}
        if last == "not_applied"
        else {"type": "not_applied", "reason": "expired"}
    )
    first = _beta_message(fallback_credit={"status": first_status})
    value = (
        None
        if last is None
        else {
            "status": {
                "type": last,
                **({"reason": "expired"} if last == "not_applied" else {}),
            }
        }
    )
    second = _beta_message(fallback_credit=value)
    expected = second.usage.model_dump()["fallback_credit"]
    total = _prepare_usage_accumulator(first, _initialize_usage(Provider.ANTHROPIC))
    update_total_usage(first, total)
    update_total_usage(second, total)
    assert total.model_dump()["fallback_credit"] == expected
    assert second.usage.model_dump()["fallback_credit"] == expected
    beta.BetaUsage.model_validate(total.model_dump())
    if last is not None:
        assert total.fallback_credit is not second.usage.fallback_credit
        status = second.usage.fallback_credit.status
        type(status).model_validate(status.model_dump())


def test_iterative_beta_usage_remains_unavailable_and_unmodified() -> None:
    from instructor.v2.core.usage import _prepare_usage_accumulator

    response = _beta_message(
        iterations=[
            {
                "type": "compaction",
                "input_tokens": 40,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        ],
        input_tokens=0,
        output_tokens=0,
    )
    before = response.usage.model_dump()
    total = _prepare_usage_accumulator(response, _initialize_usage(Provider.ANTHROPIC))
    assert not has_compatible_usage(response, total)
    update_total_usage(response, total)
    assert response.usage.model_dump() == before
    assert total.input_tokens == 0
    assert total.output_tokens == 0


def test_mixed_nonzero_stable_to_beta_usage_is_not_silently_promoted() -> None:
    from instructor.v2.core.usage import _prepare_usage_accumulator

    total = _initialize_usage(Provider.ANTHROPIC)
    total.input_tokens = 10
    response = _beta_message()
    assert _prepare_usage_accumulator(response, total) is total
    assert not has_compatible_usage(response, total)
    update_total_usage(response, total)
    assert total.input_tokens == 10


def test_beta_first_response_details_and_unknown_metadata_are_isolated() -> None:
    from instructor.v2.core.usage import _prepare_usage_accumulator

    response = _beta_message(
        audit_list=[{"counter": 7}],
        audit_number=8,
        server_tool_use={
            "web_search_requests": 1,
            "web_fetch_requests": 2,
            "audit_details": {"counter": 7},
        },
    )
    total = _prepare_usage_accumulator(response, _initialize_usage(Provider.ANTHROPIC))
    update_total_usage(response, total)
    expected = total.model_dump()
    response.usage.audit_list[0]["counter"] = 99
    response.usage.server_tool_use.audit_details["counter"] = 99
    details = response.usage.output_tokens_details
    if isinstance(details, dict):
        details["thinking_tokens"] = 99
    else:
        details.thinking_tokens = 99
    assert total.model_dump() == expected
    next_response = _beta_message(
        audit_list=[{"counter": 1}],
        audit_number=3,
        server_tool_use={
            "web_search_requests": 1,
            "web_fetch_requests": 2,
            "audit_details": {"counter": 1},
        },
    )
    update_total_usage(next_response, total)
    assert total.audit_number == 3
    assert total.audit_list == [{"counter": 1}]
    next_response.usage.audit_list[0]["counter"] = 22
    next_response.usage.server_tool_use.audit_details["counter"] = 22
    assert total.server_tool_use.audit_details == {"counter": 1}
    assert total.audit_list == [{"counter": 1}]


@pytest.mark.parametrize("key", ["_trace", "schema", "note"])
def test_beta_opaque_extras_preserve_mapping_names_and_absence(key: str) -> None:
    from instructor.v2.core.usage import _prepare_usage_accumulator

    first = _beta_message(**{key: {"value": 7}})
    total = _prepare_usage_accumulator(first, _initialize_usage(Provider.ANTHROPIC))
    update_total_usage(first, total)
    assert total.model_extra[key] == {"value": 7}
    assert total.model_dump()[key] == {"value": 7}
    first.usage.model_extra[key]["value"] = 99
    assert total.model_extra[key] == {"value": 7}
    second = _beta_message(**{key: {"value": 9}})
    update_total_usage(second, total)
    assert total.model_dump()[key] == {"value": 9}
    second.usage.model_extra[key]["value"] = 100
    assert total.model_extra[key] == {"value": 9}
    snapshot = _usage_snapshot(total)
    update_total_usage(_beta_message(), total)
    assert key not in total.model_extra
    assert key not in total.model_dump()
    assert snapshot.model_extra[key] == {"value": 9}
