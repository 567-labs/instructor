from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
import json
from typing import Any, Literal

import httpx
import pytest
import pytest_asyncio
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator

import instructor


PROVIDERS = [
    ("typesafe", "jev-latest", "/v1/systemone"),
    ("openrouter", "typesafe/jev-1.13", "/api/alpha/decisions"),
    ("openai", "gpt-6-luna", "/v1/decisions"),
]
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.parametrize("async_client", [False, True], ids=["sync", "async"]),
]


class Decision(BaseModel):
    action: Literal["allow", "review", "block"] = Field(
        description="Apply {{ policy.name }} to {{ item.text }}"
    )

    @field_validator("action")
    @classmethod
    def permitted_by_context(cls, value: str, info: ValidationInfo) -> str:
        if info.context is None:
            raise ValueError("Decision validation requires context")
        if value not in info.context["policy"]["allowed"]:
            raise ValueError(
                f"{value} is forbidden by {info.context['policy']['name']}"
            )
        return value


CONTEXT: dict[str, Any] = {
    "policy": {"name": "moderation", "allowed": ["allow", "review", "block"]},
    "item": {"text": "a disputed comment", "tags": ["public", "discussion"]},
}


def decision_response(provider: str, action: str = "review") -> dict[str, Any]:
    usage: dict[str, Any] = {"input_tokens": 17, "output_tokens": 2}
    if provider == "openrouter":
        usage["cost"] = 0.0042
    if provider == "openai":
        return {
            "model": "gpt-6-luna",
            "usage": usage,
            "answers": [
                {
                    "name": "action",
                    "type": "choice",
                    "choice": action,
                    "confidence": 0.8,
                    "probabilities": [{"value": action, "probability": 0.8}],
                }
            ],
            "metadata": {"provider": provider, "trace": ["opaque", {"attempt": 1}]},
        }
    return {
        "id": f"{provider}-request",
        "model": f"{provider}-resolved-model",
        "answers": {
            "action": {
                "type": "choice",
                "choice": action,
                "probabilities": {
                    choice: 0.8 if choice == action else 0.1
                    for choice in ("allow", "review", "block")
                },
                "confidence": 0.8,
            }
        },
        "usage": usage,
        "metadata": {"provider": provider, "trace": ["opaque", {"attempt": 1}]},
    }


def request_body(provider: str, model: str, context: dict[str, Any]) -> dict[str, Any]:
    instructions = f"Apply {context['policy']['name']} to {context['item']['text']}"
    if provider == "openai":
        return {
            "model": model,
            "input": json.dumps(context, ensure_ascii=False),
            "questions": [
                {
                    "name": "action",
                    "type": "choice",
                    "instructions": instructions,
                    "choices": [
                        {"value": choice} for choice in ("allow", "review", "block")
                    ],
                }
            ],
        }
    return {
        "model": model,
        "state": context,
        "questions": {
            "action": {
                "type": "choice",
                "instructions": instructions,
                "criteria": {choice: None for choice in ("allow", "review", "block")},
            }
        },
    }


@pytest_asyncio.fixture
async def http_client(
    async_client: bool,
) -> AsyncIterator[httpx.Client | httpx.AsyncClient]:
    if async_client:
        async with httpx.AsyncClient(trust_env=False) as http:
            yield http
    else:
        with httpx.Client(trust_env=False) as http:
            yield http


@pytest.mark.parametrize("provider,model,path", PROVIDERS)
async def test_request_timeout_override_does_not_change_borrowed_client(
    decision_endpoint, http_client, async_client, provider, model, path
):
    decision_endpoint.response = decision_response(provider)
    http_client.timeout = httpx.Timeout(9.0)
    borrowed_timeout = http_client.timeout
    observed_timeouts: list[dict[str, float | None]] = []

    def record_timeout(request: httpx.Request) -> None:
        observed_timeouts.append(deepcopy(request.extensions["timeout"]))

    async def record_async_timeout(request: httpx.Request) -> None:
        record_timeout(request)

    http_client.event_hooks["request"].append(
        record_async_timeout if async_client else record_timeout
    )
    default_client = instructor.from_provider(
        f"{provider}/{model}",
        mode=instructor.Mode.DECISIONS,
        async_client=async_client,
        api_key="timeout-contract-key",
        http_client=http_client,
        endpoint=decision_endpoint.url + path,
    )
    overridden_client = instructor.from_provider(
        f"{provider}/{model}",
        mode=instructor.Mode.DECISIONS,
        async_client=async_client,
        api_key="timeout-contract-key",
        http_client=http_client,
        endpoint=decision_endpoint.url + path,
        timeout=1.75,
    )
    unlimited_client = instructor.from_provider(
        f"{provider}/{model}",
        mode=instructor.Mode.DECISIONS,
        async_client=async_client,
        api_key="timeout-contract-key",
        http_client=http_client,
        endpoint=decision_endpoint.url + path,
        timeout=None,
    )
    expected_timeouts = []
    assert http_client.timeout is borrowed_timeout
    assert http_client.timeout.as_dict() == {
        "connect": 9.0,
        "read": 9.0,
        "write": 9.0,
        "pool": 9.0,
    }

    for client, timeout in (
        (default_client, 9.0),
        (overridden_client, 1.75),
        (unlimited_client, None),
        (default_client, 9.0),
    ):
        pending = client.create(response_model=Decision, context=deepcopy(CONTEXT))
        result = await pending if async_client else pending
        assert result.model_dump() == {"action": "review"}
        expected_timeouts.append(
            {"connect": timeout, "read": timeout, "write": timeout, "pool": timeout}
        )
        assert observed_timeouts == expected_timeouts
        assert http_client.timeout is borrowed_timeout
        assert http_client.timeout.as_dict() == {
            "connect": 9.0,
            "read": 9.0,
            "write": 9.0,
            "pool": 9.0,
        }
        assert len(decision_endpoint.calls) == len(expected_timeouts)

    assert [call["path"] for call in decision_endpoint.calls] == [path] * 4


@pytest.mark.parametrize("provider,model,path", PROVIDERS)
async def test_switch_preserves_typed_result_and_opaque_completion(
    decision_endpoint, http_client, async_client, provider, model, path
):
    context = deepcopy(CONTEXT)
    original_context = deepcopy(context)
    original_schema = deepcopy(Decision.model_json_schema())
    response = decision_response(provider)
    original_response = deepcopy(response)
    decision_endpoint.response = response
    client = instructor.from_provider(
        f"{provider}/{model}",
        mode=instructor.Mode.DECISIONS,
        async_client=async_client,
        api_key="explicit-contract-key",
        http_client=http_client,
        endpoint=decision_endpoint.url + path,
    )

    completion = client.create_with_completion(response_model=Decision, context=context)
    result, raw = await completion if async_client else completion

    assert type(result) is Decision
    assert result.model_dump() == {"action": "review"}
    assert raw == original_response
    assert context == original_context
    assert response == original_response
    assert Decision.model_json_schema() == original_schema
    assert decision_endpoint.calls == [
        {
            "method": "POST",
            "path": path,
            "authorization": "Bearer explicit-contract-key",
            "body": request_body(provider, model, original_context),
        }
    ]


@pytest.mark.parametrize("provider,model,path", PROVIDERS)
async def test_explicit_key_precedes_environment_and_is_bound_to_client(
    decision_endpoint, monkeypatch, http_client, async_client, provider, model, path
):
    for other, _, _ in PROVIDERS:
        monkeypatch.setenv(f"{other.upper()}_API_KEY", f"{other}-environment-key")
    decision_endpoint.response = decision_response(provider, "allow")
    explicit_client = instructor.from_provider(
        f"{provider}/{model}",
        mode=instructor.Mode.DECISIONS,
        async_client=async_client,
        api_key=f"{provider}-explicit-key",
        http_client=http_client,
        endpoint=decision_endpoint.url + path,
    )
    environment_client = instructor.from_provider(
        f"{provider}/{model}",
        mode=instructor.Mode.DECISIONS,
        async_client=async_client,
        http_client=http_client,
        endpoint=decision_endpoint.url + path,
    )
    for other, _, _ in PROVIDERS:
        monkeypatch.setenv(f"{other.upper()}_API_KEY", f"{other}-rotated-key")

    for client in (explicit_client, environment_client, explicit_client):
        pending = client.create(response_model=Decision, context=deepcopy(CONTEXT))
        result = await pending if async_client else pending
        assert result.model_dump() == {"action": "allow"}

    assert [call["authorization"] for call in decision_endpoint.calls] == [
        f"Bearer {provider}-explicit-key",
        f"Bearer {provider}-environment-key",
        f"Bearer {provider}-explicit-key",
    ]
    assert [call["path"] for call in decision_endpoint.calls] == [path] * 3
    assert [call["body"]["model"] for call in decision_endpoint.calls] == [model] * 3


@pytest.mark.parametrize(
    "provider_order", [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
)
async def test_alternating_clients_isolate_context_auth_and_response(
    decision_endpoint, http_client, async_client, provider_order
):
    clients = [
        instructor.from_provider(
            f"{provider}/{model}",
            mode=instructor.Mode.DECISIONS,
            async_client=async_client,
            api_key=f"{provider}-isolated-key",
            http_client=http_client,
            endpoint=decision_endpoint.url + path,
        )
        for provider, model, path in PROVIDERS
    ]
    schema_before = deepcopy(Decision.model_json_schema())
    retained = []
    for sequence, index in enumerate(list(provider_order) * 2):
        provider, model, path = PROVIDERS[index]
        action = ["review", "allow", "block", "review"][sequence]
        context = deepcopy(CONTEXT)
        context["policy"]["name"] = f"policy-{sequence}"
        context["policy"]["allowed"] = [action]
        context["item"]["tags"].append(f"request-{sequence}")
        context_before = deepcopy(context)
        response = decision_response(provider, action)
        response["id"] = f"request-{sequence}"
        response_before = deepcopy(response)
        decision_endpoint.response = response

        pending = clients[index].create_with_completion(
            response_model=Decision, context=context
        )
        result, raw = await pending if async_client else pending

        assert result.model_dump() == {"action": action}
        assert raw == response_before
        assert context == context_before
        assert response == response_before
        assert len(decision_endpoint.calls) == sequence + 1
        call = decision_endpoint.calls[sequence]
        assert call["method"] == "POST"
        assert call["path"] == path
        assert call["authorization"] == f"Bearer {provider}-isolated-key"
        assert call["body"]["model"] == model
        assert call["body"] == request_body(provider, model, context_before)
        retained.append((result, raw, action, response_before))

    for result, raw, action, response_before in retained:
        assert result.model_dump() == {"action": action}
        assert raw == response_before
    assert Decision.model_json_schema() == schema_before


@pytest.mark.parametrize("provider,model,path", PROVIDERS)
async def test_context_validation_failure_is_specific_and_next_request_recovers(
    decision_endpoint, http_client, async_client, provider, model, path
):
    decision_endpoint.response = decision_response(provider)
    response_before = deepcopy(decision_endpoint.response)
    context = deepcopy(CONTEXT)
    context["policy"] = {"name": "restricted", "allowed": ["allow"]}
    context_before = deepcopy(context)
    client = instructor.from_provider(
        f"{provider}/{model}",
        mode=instructor.Mode.DECISIONS,
        async_client=async_client,
        api_key="validation-contract-key",
        http_client=http_client,
        endpoint=decision_endpoint.url + path,
    )

    with pytest.raises(
        ValidationError, match="review is forbidden by restricted"
    ) as exc:
        pending = client.create(response_model=Decision, context=context)
        if async_client:
            await pending
    errors = exc.value.errors(include_url=False)
    assert len(errors) == 1
    assert errors[0]["loc"] == ("action",)
    assert errors[0]["type"] == "value_error"
    assert errors[0]["input"] == "review"
    assert context == context_before
    assert decision_endpoint.response == response_before
    assert len(decision_endpoint.calls) == 1
    assert decision_endpoint.calls[0]["body"] == request_body(
        provider, model, context_before
    )

    allowed_context = deepcopy(context)
    allowed_context["policy"]["allowed"] = ["review"]
    pending = client.create(response_model=Decision, context=allowed_context)
    result = await pending if async_client else pending
    assert result.model_dump() == {"action": "review"}
    assert len(decision_endpoint.calls) == 2
    assert decision_endpoint.calls[1]["body"] == request_body(
        provider, model, allowed_context
    )
    assert context == context_before


@pytest.mark.parametrize("provider,model,path", PROVIDERS)
@pytest.mark.parametrize(
    "response",
    [None, [], "not an answers object", {"answers": []}],
    ids=["null", "array", "string", "answers-array"],
)
async def test_malformed_json_envelopes_fail_without_typed_fallback(
    decision_endpoint, http_client, async_client, provider, model, path, response
):
    decision_endpoint.response = response
    client = instructor.from_provider(
        f"{provider}/{model}",
        mode=instructor.Mode.DECISIONS,
        async_client=async_client,
        api_key="malformed-contract-key",
        http_client=http_client,
        endpoint=decision_endpoint.url + path,
    )

    with pytest.raises(
        ValueError,
        match="[Dd]ecision response must contain an answers (object|array)|answer count",
    ):
        pending = client.create(response_model=Decision, context=deepcopy(CONTEXT))
        if async_client:
            await pending
    assert len(decision_endpoint.calls) == 1
    assert decision_endpoint.calls[0]["path"] == path
    assert decision_endpoint.calls[0]["body"]["model"] == model


@pytest.mark.parametrize("provider,model,path", PROVIDERS)
async def test_http_failure_preserves_response_and_client_recovers(
    decision_endpoint, http_client, async_client, provider, model, path
):
    error_body = {"error": {"code": "upstream_unavailable", "retryable": True}}
    decision_endpoint.status = 502
    decision_endpoint.response = deepcopy(error_body)
    client = instructor.from_provider(
        f"{provider}/{model}",
        mode=instructor.Mode.DECISIONS,
        async_client=async_client,
        api_key="http-error-contract-key",
        http_client=http_client,
        endpoint=decision_endpoint.url + path,
    )

    with pytest.raises(httpx.HTTPStatusError) as exc:
        pending = client.create(response_model=Decision, context=deepcopy(CONTEXT))
        if async_client:
            await pending
    assert exc.value.response.status_code == 502
    assert exc.value.response.json() == error_body
    assert str(exc.value.request.url) == decision_endpoint.url + path
    assert exc.value.request.method == "POST"
    assert len(decision_endpoint.calls) == 1

    decision_endpoint.status = 200
    decision_endpoint.response = decision_response(provider, "block")
    pending = client.create(response_model=Decision, context=deepcopy(CONTEXT))
    result = await pending if async_client else pending
    assert result.model_dump() == {"action": "block"}
    assert len(decision_endpoint.calls) == 2
    assert [call["authorization"] for call in decision_endpoint.calls] == [
        "Bearer http-error-contract-key",
        "Bearer http-error-contract-key",
    ]


@pytest.mark.parametrize("provider,model,path", PROVIDERS)
async def test_invalid_http_json_raises_decode_error_without_fallback(
    decision_endpoint, http_client, async_client, provider, model, path
):
    decision_endpoint.response = decision_response(provider)
    decision_endpoint.response_body = b'{"answers":'
    context = deepcopy(CONTEXT)
    context_before = deepcopy(context)
    client = instructor.from_provider(
        f"{provider}/{model}",
        mode=instructor.Mode.DECISIONS,
        async_client=async_client,
        api_key="invalid-json-contract-key",
        http_client=http_client,
        endpoint=decision_endpoint.url + path,
    )

    with pytest.raises(json.JSONDecodeError) as exc:
        pending = client.create(response_model=Decision, context=context)
        if async_client:
            await pending
    assert exc.value.doc == '{"answers":'
    assert exc.value.pos == len(exc.value.doc)
    assert context == context_before
    assert len(decision_endpoint.calls) == 1
    call = decision_endpoint.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == path
    assert call["authorization"] == "Bearer invalid-json-contract-key"
    assert call["body"]["model"] == model
    assert call["body"] == request_body(provider, model, context_before)
