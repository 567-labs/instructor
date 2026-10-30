from __future__ import annotations

from copy import deepcopy
import json
from typing import Annotated, Literal

import httpx
import pytest
from pydantic import BaseModel, Field

import instructor
from instructor.decisions import Choice, Noul, Question, Score


class Decision(BaseModel):
    action: Literal["allow", "review"] = Field(description="Apply {{ policy }}")
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


@pytest.mark.asyncio
@pytest.mark.parametrize("async_client", [False, True])
async def test_openai_decisions_round_trip(decision_endpoint, async_client):
    raw = completion()
    decision_endpoint.response = deepcopy(raw)
    context = {"policy": "moderation", "text": "A disputed comment"}
    original = deepcopy(context)
    http = (
        httpx.AsyncClient(trust_env=False)
        if async_client
        else httpx.Client(trust_env=False)
    )
    client = instructor.from_provider(
        "openai/gpt-6-luna",
        mode=instructor.Mode.DECISIONS,
        async_client=async_client,
        api_key="local-contract-key",
        http_client=http,
        endpoint=decision_endpoint.url + "/v1/decisions",
    )
    try:
        pending = client.create_with_completion(
            response_model=Decision, context=context
        )
        result, returned = await pending if async_client else pending
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
                "input": json.dumps(context, ensure_ascii=False),
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
        if async_client:
            await client.close()
        else:
            client.close()
        assert not http.is_closed
    finally:
        if isinstance(http, httpx.AsyncClient):
            await http.aclose()
        else:
            http.close()


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
