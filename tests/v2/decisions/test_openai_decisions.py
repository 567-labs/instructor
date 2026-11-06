from __future__ import annotations

from contextlib import AsyncExitStack
from copy import deepcopy
import json
from typing import Annotated, Literal
from unittest.mock import Mock

import httpx
import openai
import pytest
from pydantic import BaseModel, Field

import instructor
from instructor.decisions import Choice, Noul, Question, Score
from instructor.decisions.client import AsyncDecisionsClient


class Decision(BaseModel):
    action: Literal["allow", "review"] = Field(
        description="Apply {{ policy | default('the policy') }}"
    )
    probability: Annotated[
        float,
        Noul(instructions="Is this unsafe?", when_true="Unsafe", when_false="Safe"),
    ]
    severity: Annotated[
        float,
        Score(
            instructions="How severe?", levels=["Low", "Medium", "High"], scale=(0, 10)
        ),
    ]


def completion() -> dict:
    return {
        "model": "gpt-6-luna",
        "answers": [
            {
                "name": "action",
                "type": "choice",
                "choice": "review",
                "confidence": 0.9,
                "probabilities": [{"value": "review", "probability": 0.9}],
            },
            {"name": "probability", "type": "predicate", "probability": 0.8},
            {"name": "severity", "type": "score", "score": 1.4, "confidence": 0.7},
        ],
        "usage": {"input_tokens": 12, "total_tokens": 12},
    }


def test_supplied_sdk_prefers_decisions_resource_and_keeps_raw_fields(monkeypatch):
    raw = {**completion(), "provider_metadata": {"version": 3}}
    resource = Mock()
    resource.with_raw_response.create.return_value.http_response = httpx.Response(
        200, json=raw
    )
    monkeypatch.setattr(openai.OpenAI, "decisions", resource, raising=False)
    with openai.OpenAI(
        api_key="local-contract-key",
        http_client=openai.DefaultHttpxClient(trust_env=False),
    ) as sdk:
        with instructor.from_openai(sdk, mode=instructor.Mode.DECISIONS) as client:
            result, returned = client.create_with_completion(
                response_model=Decision, input="Evaluate this ticket."
            )
        assert not sdk.is_closed()
    assert result.severity == 7.0 and returned == raw
    request = resource.with_raw_response.create.call_args.kwargs
    assert request["input"] == "Evaluate this ticket."
    assert request["questions"][0]["instructions"] == "Apply the policy"
    assert [q["type"] for q in request["questions"]] == ["choice", "predicate", "score"]


def test_supplied_sdk_preserves_caller_settings_and_does_not_retry(decision_endpoint):
    decision_endpoint.status = 503
    with openai.OpenAI(
        api_key="local-contract-key",
        base_url=decision_endpoint.url + "/v1",
        http_client=openai.DefaultHttpxClient(trust_env=False),
        max_retries=2,
    ) as sdk:
        timeout = sdk.timeout
        with instructor.from_openai(
            sdk, mode=instructor.Mode.DECISIONS, timeout=5.0
        ) as client:
            with pytest.raises(openai.APIStatusError):
                client.create(response_model=Decision, input="Evaluate this ticket.")
        assert sdk.max_retries == 2 and sdk.timeout == timeout
        assert not sdk.is_closed()
    assert len(decision_endpoint.calls) == 1


def test_jev_rejects_openai_input_before_sending(decision_endpoint):
    with (
        httpx.Client(trust_env=False) as http,
        instructor.from_provider(
            "typesafe/jev-latest",
            mode=instructor.Mode.DECISIONS,
            api_key="local-contract-key",
            http_client=http,
            endpoint=decision_endpoint.url,
        ) as client,
    ):
        with pytest.raises(TypeError, match="only for OpenAI"):
            client.create(response_model=Decision, input="Evaluate this ticket.")
        with pytest.raises(TypeError, match="context must be a dictionary"):
            client.create(response_model=Decision)
    assert decision_endpoint.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("async_client", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("use_sdk", [False, True], ids=["provider", "sdk"])
@pytest.mark.parametrize(
    "native_input",
    [
        None,
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_json", "data": {"ticket": "duplicate charge"}},
                    {
                        "type": "input_image",
                        "image_url": "https://example.test/policy.png",
                    },
                ],
            }
        ],
    ],
    ids=["context", "content-parts"],
)
async def test_openai_decisions_round_trip(
    decision_endpoint, async_client, use_sdk, native_input
):
    raw = completion()
    decision_endpoint.response = deepcopy(raw)
    context = {"policy": "moderation", "text": "A disputed comment"}
    original = deepcopy(context)
    async with AsyncExitStack() as stack:
        if use_sdk and async_client:
            transport = openai.DefaultAsyncHttpxClient(trust_env=False)
            sdk = await stack.enter_async_context(
                openai.AsyncOpenAI(
                    api_key="local-contract-key",
                    base_url=decision_endpoint.url + "/v1",
                    http_client=transport,
                )
            )
            client = instructor.from_openai(sdk, mode=instructor.Mode.DECISIONS)
        elif use_sdk:
            transport = openai.DefaultHttpxClient(trust_env=False)
            sdk = stack.enter_context(
                openai.OpenAI(
                    api_key="local-contract-key",
                    base_url=decision_endpoint.url + "/v1",
                    http_client=transport,
                )
            )
            client = instructor.from_openai(sdk, mode=instructor.Mode.DECISIONS)
        else:
            transport = (
                await stack.enter_async_context(httpx.AsyncClient(trust_env=False))
                if async_client
                else stack.enter_context(httpx.Client(trust_env=False))
            )
            client = instructor.from_provider(
                "openai/gpt-6-luna",
                mode=instructor.Mode.DECISIONS,
                async_client=async_client,
                api_key="local-contract-key",
                http_client=transport,
                endpoint=decision_endpoint.url + "/v1/decisions",
            )
        if isinstance(client, AsyncDecisionsClient):
            async with client:
                result, returned = await client.create_with_completion(
                    response_model=Decision, context=context, input=native_input
                )
        else:
            with client:
                result, returned = client.create_with_completion(
                    response_model=Decision, context=context, input=native_input
                )
        assert not transport.is_closed
        assert result.model_dump() == {
            "action": "review",
            "probability": 0.8,
            "severity": 7.0,
        }
        assert returned == raw
        assert context == original
        assert decision_endpoint.calls[0] == {
            "method": "POST",
            "path": "/v1/decisions",
            "authorization": "Bearer local-contract-key",
            "body": {
                "model": "gpt-6-luna",
                "input": json.dumps(context, ensure_ascii=False)
                if native_input is None
                else native_input,
                "questions": [
                    {
                        "name": "action",
                        "type": "choice",
                        "instructions": "Apply moderation",
                        "choices": [{"value": "allow"}, {"value": "review"}],
                    },
                    {
                        "name": "probability",
                        "type": "predicate",
                        "instructions": 'Is this unsafe?\nCriteria: {"true": "Unsafe", "false": "Safe"}',
                    },
                    {
                        "name": "severity",
                        "type": "score",
                        "instructions": "How severe?",
                        "levels": [
                            {"label": str(i), "description": value}
                            for i, value in enumerate(["Low", "Medium", "High"])
                        ],
                    },
                ],
            },
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize(
    "problem",
    [
        "count",
        "name",
        "type",
        "refusal",
        "probability",
        "score",
        "choice",
        "envelope",
        "item",
    ],
)
async def test_openai_invalid_answers_fail(decision_endpoint, async_client, problem):
    raw = completion()
    if problem == "count":
        raw["answers"].pop()
    elif problem == "name":
        raw["answers"][0]["name"] = "other"
    elif problem in ("type", "refusal"):
        raw["answers"][0]["type"] = "score" if problem == "type" else "refusal"
    elif problem == "probability":
        raw["answers"][1]["probability"] = 1.1
    elif problem == "score":
        raw["answers"][2]["score"] = 3
    elif problem == "choice":
        raw["answers"][0]["choice"] = "unknown"
    elif problem == "item":
        raw["answers"][0] = None
    else:
        raw["answers"] = {}
    decision_endpoint.response = raw
    client = instructor.from_provider(
        "openai/gpt-6-luna",
        mode=instructor.Mode.DECISIONS,
        async_client=async_client,
        api_key="local-contract-key",
        endpoint=decision_endpoint.url + "/v1/decisions",
    )
    try:
        with pytest.raises(ValueError):
            pending = client.create(
                response_model=Decision, context={"policy": "moderation"}
            )
            if async_client:
                await pending
        assert len(decision_endpoint.calls) == 1
    finally:
        if async_client:
            await client.close()
        else:
            client.close()


def test_openai_structured_question_metadata_is_preserved():
    from instructor.decisions.schema import build_questions
    from instructor.decisions.openai import request_body

    class Structured(BaseModel):
        action: Annotated[
            Literal["allow", "review"],
            Choice(
                description={"policy": "review contested cases"}, examples=["dispute"]
            ),
            Question(instructions={"question": "Which action?"}),
        ]
        probability: Annotated[float, Noul(instructions="Is it unsafe?")]

    body = request_body("gpt-6-luna", build_questions(Structured, {}), {})
    choice, predicate = body["questions"]
    assert json.loads(choice["instructions"]) == {"question": "Which action?"}
    assert json.loads(choice["choices"][0]["description"]) == {
        "meaning": {"policy": "review contested cases"},
        "examples": ["dispute"],
    }
    assert predicate == {
        "name": "probability",
        "type": "predicate",
        "instructions": "Is it unsafe?",
    }
