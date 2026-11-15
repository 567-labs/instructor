"""Usage accounting across provider SDK usage shapes.

One table covers `update_total_usage` for every supported SDK shape: each row
accumulates real SDK responses into a running total and checks the totals and
the totals echoed back onto the latest response. Known gaps are strict xfails
that assert the desired behavior, so a fix flips them to XPASS and fails CI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, NamedTuple

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response
from pydantic import BaseModel

from instructor.v2.core.providers import Provider
from instructor.v2.core.retry import (
    _initialize_usage,
    _usage_snapshot,
    _usage_total_tokens,
)
from instructor.v2.core.usage import (
    _accumulate_models,
    has_compatible_usage,
    update_total_usage,
)
from instructor.v2.providers.openai.handlers import OpenAIJSONHandler
from tests._streams import consume


def chat(usage: CompletionUsage | dict[str, Any]) -> ChatCompletion:
    if isinstance(usage, dict):
        usage = CompletionUsage.model_validate(usage)
    return ChatCompletion(
        id="usage",
        created=0,
        model="usage",
        object="chat.completion",
        choices=[],
        usage=usage,
    )


def responses_api(
    input_tokens: int, output_tokens: int, cached: int, reasoning: int
) -> Response:
    return Response.model_validate(
        {
            "id": "usage",
            "created_at": 0,
            "object": "response",
            "model": "usage",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_tokens_details": {
                    "cached_tokens": cached,
                    "cache_write_tokens": 0,
                },
                "output_tokens_details": {"reasoning_tokens": reasoning},
            },
        }
    )


def anthropic_message(usage: dict[str, Any]) -> Any:
    types = pytest.importorskip("anthropic.types")
    return types.Message.model_validate(
        {
            "id": "usage",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "usage",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }
    )


class LiteLLMUsage(CompletionUsage):
    """Provider subclass of the OpenAI usage type (as LiteLLM returns)."""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _openai_total(**extra: Any) -> Callable[[], Any]:
    return lambda: CompletionUsage.model_validate(
        {"completion_tokens": 4, "prompt_tokens": 6, "total_tokens": 10, **extra}
    )


def _initial(provider: Provider) -> Callable[[], Any]:
    return lambda: _initialize_usage(provider)


DETAILED_TOTAL = {
    "completion_tokens_details": {
        "accepted_prediction_tokens": 3,
        "audio_tokens": 1,
        "reasoning_tokens": 2,
        "rejected_prediction_tokens": 4,
    },
    "prompt_tokens_details": {
        "audio_tokens": 3,
        "cached_tokens": 4,
        "cache_write_tokens": 5,
        "vendor_write_tokens": 2,
    },
    "future_tokens": 4,
    "future_cost": 1.5,
    "service_tier": "scale",
}
DETAILED_RESPONSE = {
    "completion_tokens": 7,
    "prompt_tokens": 11,
    "total_tokens": 18,
    "completion_tokens_details": {
        "accepted_prediction_tokens": 7,
        "audio_tokens": 5,
        "reasoning_tokens": 8,
        "rejected_prediction_tokens": 9,
    },
    "prompt_tokens_details": {
        "audio_tokens": 13,
        "cached_tokens": 21,
        "cache_write_tokens": 23,
        "vendor_write_tokens": 7,
    },
    "future_tokens": 7,
    "future_cost": 2.25,
}
REPEATED_OPENAI = {
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
ANTHROPIC_CACHE = {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_read_input_tokens": 200,
    "cache_creation_input_tokens": 30,
    "cache_creation": {
        "ephemeral_5m_input_tokens": 20,
        "ephemeral_1h_input_tokens": 10,
    },
    "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1},
}


class UsageCase(NamedTuple):
    total: Callable[[], Any]
    responses: Callable[[], list[Any]]
    tokens: int
    expected: dict[str, Any]
    # Usage echoed onto the latest response; defaults to `expected`.
    echo: dict[str, Any] | None = None


USAGE_CASES = {
    "openai-details-and-extras": UsageCase(
        _openai_total(**DETAILED_TOTAL),
        lambda: [chat(DETAILED_RESPONSE)],
        28,
        {
            "completion_tokens": 11,
            "prompt_tokens": 17,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 10,
                "audio_tokens": 6,
                "reasoning_tokens": 10,
                "rejected_prediction_tokens": 13,
            },
            "prompt_tokens_details": {
                "audio_tokens": 16,
                "cached_tokens": 25,
                "cache_write_tokens": 28,
                "vendor_write_tokens": 9,
            },
            "future_tokens": 11,
            "future_cost": 3.75,
            # Total-only non-numeric fields are carried onto the response.
            "service_tier": "scale",
        },
    ),
    "openai-response-non-numeric-wins": UsageCase(
        _openai_total(service_tier="scale"),
        lambda: [
            chat(
                {
                    "completion_tokens": 5,
                    "prompt_tokens": 6,
                    "total_tokens": 11,
                    "service_tier": "default",
                }
            )
        ],
        21,
        {"service_tier": "default"},
    ),
    "openai-subsets-not-double-counted": UsageCase(
        _initial(Provider.OPENAI),
        lambda: [chat(REPEATED_OPENAI), chat(REPEATED_OPENAI)],
        300,
        {
            "prompt_tokens_details": {"cached_tokens": 160},
            "completion_tokens_details": {"reasoning_tokens": 60},
            "is_billable": True,
        },
    ),
    "openai-echo-running-total": UsageCase(
        _initial(Provider.OPENAI),
        lambda: [
            chat(
                CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            ),
            chat(
                CompletionUsage(prompt_tokens=20, completion_tokens=7, total_tokens=27)
            ),
        ],
        42,
        # The total's zeroed details are copied onto responses that lacked them.
        {"total_tokens": 42, "prompt_tokens_details": {"cached_tokens": 0}},
    ),
    "openai-usage-subclass": UsageCase(
        _openai_total(),
        lambda: [
            chat(LiteLLMUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7))
        ],
        17,
        {"prompt_tokens": 9, "completion_tokens": 8},
    ),
    "openai-responses-api": UsageCase(
        _initial(Provider.OPENAI),
        lambda: [responses_api(10, 5, 8, 2), responses_api(20, 7, 5, 3)],
        42,
        {
            "prompt_tokens": 30,
            "completion_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 13},
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
        {
            "total_tokens": 42,
            "input_tokens": 30,
            "output_tokens": 12,
            "input_tokens_details": {"cached_tokens": 13},
            "output_tokens_details": {"reasoning_tokens": 5},
        },
    ),
    "anthropic-plain": UsageCase(
        _initial(Provider.ANTHROPIC),
        lambda: [anthropic_message({"input_tokens": 10, "output_tokens": 5})],
        15,
        {"input_tokens": 10, "output_tokens": 5},
    ),
    "anthropic-thinking-details": UsageCase(
        _initial(Provider.ANTHROPIC),
        lambda: [
            anthropic_message(
                {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "output_tokens_details": {"thinking_tokens": 12},
                }
            )
            for _ in range(2)
        ],
        60,
        {"output_tokens": 40, "output_tokens_details": {"thinking_tokens": 24}},
    ),
    "anthropic-cache-ttl-and-tools": UsageCase(
        _initial(Provider.ANTHROPIC),
        lambda: [anthropic_message(ANTHROPIC_CACHE) for _ in range(2)],
        760,
        {
            "cache_creation": {"ephemeral_1h_input_tokens": 20},
            "server_tool_use": {"web_search_requests": 4},
        },
    ),
}


def assert_subset(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if isinstance(value, dict):
            assert_subset(actual[key], value)
        else:
            assert actual[key] == value, key


@pytest.mark.parametrize("case", USAGE_CASES.values(), ids=USAGE_CASES.keys())
def test_update_total_usage(case: UsageCase) -> None:
    responses = case.responses()
    usage_types = [type(response.usage) for response in responses]
    total = case.total()
    for response in responses:
        assert has_compatible_usage(response, total)
        assert update_total_usage(response, total) is response
    last = responses[-1]

    assert [type(response.usage) for response in responses] == usage_types
    assert _usage_total_tokens(total) == case.tokens
    assert_subset(total.model_dump(), case.expected)
    assert_subset(last.usage.model_dump(), case.echo or case.expected)
    if type(last.usage) is type(total):
        # Echoed nested details are copies, never aliases of the running total.
        for name in type(total).model_fields:
            if isinstance(getattr(total, name), BaseModel):
                assert getattr(last.usage, name) is not getattr(total, name)


def test_update_total_usage_ignores_missing_or_unsupported_usage() -> None:
    total = _initialize_usage(Provider.OPENAI)
    assert update_total_usage(None, total) is None
    unsupported = _initialize_usage(Provider.ANTHROPIC)
    response = chat(
        CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    )
    assert not has_compatible_usage(response, unsupported)
    assert update_total_usage(response, unsupported) is response
    assert response.usage is not None and response.usage.total_tokens == 2


@pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.ANTHROPIC])
def test_usage_snapshot_is_isolated_from_later_attempts(provider: Provider) -> None:
    if provider is Provider.ANTHROPIC:
        response = anthropic_message({"input_tokens": 10, "output_tokens": 5})
    else:
        response = chat(
            CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )
    total = _initialize_usage(provider)
    update_total_usage(response, total)
    snapshot = _usage_snapshot(total)
    assert snapshot.model_dump() == total.model_dump()
    update_total_usage(response.model_copy(deep=True), total)
    assert _usage_total_tokens(total) == 30
    assert _usage_total_tokens(snapshot) == 15


def test_accumulate_models_follows_recursive_sdk_shape_changes() -> None:
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
        total=2, details=Details(counters=Counters(billed=4, label="new"), region="us")
    )
    _accumulate_models(first, total)
    assert total.details == Details(
        counters=Counters(billed=4, label="new"), region="us"
    )

    second = Usage(total=3)
    _accumulate_models(second, total)
    assert second.total == 5
    assert second.details == total.details
    assert second.details is not total.details


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [(0, 0, 0), (None, 6, 6), (12, None, None), (True, 6, 6), (12, False, False)],
)
def test_anthropic_dictionary_thinking_preserves_non_counts(
    previous: int | None, current: int | None, expected: int | None
) -> None:
    types = pytest.importorskip("anthropic.types")
    if "output_tokens_details" in types.Usage.model_fields:
        pytest.skip("Invalid extra-dictionary payloads apply only to untyped SDKs.")
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


# Known limitations. Each strict xfail asserts the desired behavior.


@pytest.mark.xfail(strict=True, reason="nested dict usage keeps only last attempt")
def test_nested_dictionary_cost_accumulates() -> None:
    total = _initialize_usage(Provider.OPENAI)
    for cost in [0.25, 0.75]:
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": cost,
            "cost_details": {"upstream_inference_cost": cost},
        }
        update_total_usage(chat(usage), total)
    assert total.model_extra is not None
    assert total.model_extra["cost"] == 1.0
    assert total.model_extra["cost_details"]["upstream_inference_cost"] == 1.0


def _groq_completion() -> tuple[Any, Any]:
    groq = pytest.importorskip("groq.types.chat")
    response = groq.ChatCompletion.model_validate(
        {
            "id": "usage",
            "created": 0,
            "object": "chat.completion",
            "model": "usage",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "queue_time": 0.1,
            },
        }
    )
    return response, _initialize_usage(Provider.GROQ)


def _anthropic_beta_message() -> tuple[Any, Any]:
    beta = pytest.importorskip("anthropic.types.beta")
    response = beta.BetaMessage.model_validate(
        {
            "id": "usage",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "usage",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    return response, _initialize_usage(Provider.ANTHROPIC)


@pytest.mark.xfail(strict=True, reason="SDK usage type is not recognized")
@pytest.mark.parametrize("build", [_groq_completion, _anthropic_beta_message])
def test_sdk_usage_variant_is_recognized(
    build: Callable[[], tuple[Any, Any]],
) -> None:
    response, total = build()
    assert has_compatible_usage(response, total)


@pytest.mark.xfail(strict=True, reason="CLI prices gpt-4o-mini as gpt-4o")
def test_cli_mini_model_has_its_own_price() -> None:
    from instructor.cli.usage import get_model_cost

    # Deliberately exercise a runtime model name omitted from the legacy literal.
    mini_cost = get_model_cost("gpt-4o-mini")  # ty: ignore[invalid-argument-type]
    assert mini_cost is not get_model_cost("gpt-4o")


@pytest.mark.xfail(strict=True, reason="one usage copy is shared by all handlers")
def test_usage_event_copy_is_isolated_between_handlers() -> None:
    from instructor.v2.core.hooks import Hooks

    hooks = Hooks()
    observed: list[int] = []

    def change_event(usage: CompletionUsage) -> None:
        usage.total_tokens = 999

    hooks.on("completion:usage", change_event)
    hooks.on("completion:usage", lambda usage: observed.append(usage.total_tokens))
    total = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    hooks.emit_completion_usage(_usage_snapshot(total))
    assert total.total_tokens == 15
    assert observed == [15]


# Current characterizations: usage is not surfaced by these paths.


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_stream_extractor_skips_usage_only_chunk(is_async: bool) -> None:
    chunk = ChatCompletionChunk(
        id="usage",
        created=0,
        model="usage",
        object="chat.completion.chunk",
        choices=[],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    handler = OpenAIJSONHandler()
    extracted = await consume(handler, [chunk], is_async, "extract_streaming_json")
    assert extracted == []


def test_batch_success_omits_per_item_usage() -> None:
    from instructor.batch import BatchProcessor, BatchSuccess

    class Item(BaseModel):
        value: int

    raw = {
        "custom_id": "item",
        "response": {
            "body": {
                "choices": [{"message": {"content": '{"value": 1}'}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        },
    }
    result = BatchProcessor("openai/usage", Item).parse_results(json.dumps(raw))[0]
    assert isinstance(result, BatchSuccess)
    assert set(result.model_dump()) == {"custom_id", "result", "success"}
    assert not hasattr(result.result, "_raw_response")


def test_local_cache_replays_raw_usage_without_total() -> None:
    from instructor.cache import AutoCache, load_cached_response, store_cached_response
    from instructor.v2.core.retry import _finalize_parsed_response

    class Item(BaseModel):
        value: int

    raw = chat(CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    original = _finalize_parsed_response(Item(value=1), raw, raw.usage)
    cache = AutoCache()
    store_cached_response(cache, "usage", original)
    restored = load_cached_response(cache, "usage", Item)
    assert restored._raw_response.usage.total_tokens == 15
    assert not isinstance(restored._raw_response, ChatCompletion)
    assert not hasattr(restored, "_total_usage")
