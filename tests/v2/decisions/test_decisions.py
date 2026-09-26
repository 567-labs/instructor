from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationInfo, field_validator

import instructor
from instructor.decisions import Choice, Choices, Level, Noul, Question, Score


class Category(Choices):
    """Which policy category?"""

    HARASSMENT = Choice(
        description={"definition": "Personal attacks"}, examples=["Go away"]
    )
    NONE = Choice(value="safe")


class Decision(BaseModel):
    category: Category
    violation: Annotated[
        float,
        Noul(
            instructions={
                "question": "Does {{ post.kind }} violate policy?",
                "rules": ["Be civil"],
            },
            when_true="Abuse",
            when_false="Disagreement",
            examples=[("attack", True), ("debate", False)],
        ),
    ]
    severity: Annotated[
        float,
        Score(
            instructions="How severe?",
            scale=(0, 10),
            levels=[
                Level("None", examples=["Hello"]),
                Level("Insult"),
                Level("Threat"),
            ],
            examples=[("Threatening", 2)],
        ),
    ]
    format: Literal["text", "image"] = Field(description="What format?")
    target: (
        Annotated[Literal["person"], Choice(description="A person")]
        | Annotated[Literal["group"], Choice(description="A group")]
    ) = Field(description="Who is targeted?")

    @field_validator("severity")
    @classmethod
    def context_available(cls, value: float, info: ValidationInfo) -> float:
        assert info.context is not None
        assert info.context["post"]["kind"] == "comment"
        return value


RAW: dict[str, Any] = {
    "model": "jev-1.13",
    "answers": {
        "category": {
            "type": "choice",
            "choice": "harassment",
            "probabilities": {"harassment": 0.9, "safe": 0.1},
            "confidence": 0.8,
        },
        "violation": {"type": "noul", "noul": 0.9},
        "severity": {
            "type": "score",
            "score": 1.4,
            "probabilities": {"0": 0.0, "1": 0.6, "2": 0.4},
            "legend": {"0": "None", "1": "Insult", "2": "Threat"},
            "confidence": 0.5,
        },
        "format": {"type": "choice", "choice": "text"},
        "target": {"type": "choice", "choice": "person"},
    },
}
CONTEXT = {"post": {"kind": "comment", "text": "Go away"}, "policy": ["Be civil"]}


@pytest.mark.parametrize(
    "provider,model,url",
    [
        (
            "openrouter",
            "typesafe/jev-1.13",
            "https://openrouter.ai/api/alpha/decisions",
        ),
        ("typesafe", "jev-latest", "https://api.typesafe.ai/v1/systemone"),
    ],
)
def test_request_and_typed_response(provider, model, url):
    def handler(request):
        assert str(request.url) == url
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == model
        assert body["state"] == CONTEXT
        questions = body["questions"]
        assert questions["category"] == {
            "type": "choice",
            "instructions": "Which policy category?",
            "criteria": {
                "harassment": {
                    "meaning": {"definition": "Personal attacks"},
                    "examples": ["Go away"],
                },
                "safe": None,
            },
        }
        assert questions["violation"] == {
            "type": "noul",
            "instructions": {
                "question": "Does comment violate policy?",
                "rules": ["Be civil"],
            },
            "criteria": {
                "true": {"meaning": "Abuse", "examples": ["attack"]},
                "false": {"meaning": "Disagreement", "examples": ["debate"]},
            },
        }
        assert questions["severity"]["criteria"] == [
            {"meaning": "None", "examples": ["Hello"]},
            "Insult",
            {"meaning": "Threat", "examples": ["Threatening"]},
        ]
        assert questions["target"]["criteria"] == {
            "person": "A person",
            "group": "A group",
        }
        return httpx.Response(200, json=RAW)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = instructor.from_provider(
            f"{provider}/{model}",
            mode=instructor.Mode.DECISIONS,
            api_key="test-key",
            http_client=http,
        )
        result, raw = client.create_with_completion(
            response_model=Decision, context=CONTEXT
        )
        assert result.category is Category.HARASSMENT
        assert result.violation == 0.9
        assert result.severity == 7
        assert result.target == "person"
        assert raw == RAW
        client.close()
        assert not http.is_closed


def test_async_client_and_errors():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json=RAW)
            )
        ) as http:
            client = instructor.from_provider(
                "openrouter/typesafe/jev-1.13",
                mode=instructor.Mode.DECISIONS,
                async_client=True,
                api_key="key",
                http_client=http,
            )
            result = await client.create(response_model=Decision, context=CONTEXT)
            assert result.severity == 7
            await client.close()
            assert not http.is_closed

    asyncio.run(run())


def test_plain_enum_structured_question_and_alias():
    class Action(str, Enum):
        ALLOW = "allow"
        REVIEW = "review"

    class Model(BaseModel):
        action: Annotated[
            Action, Question(instructions={"question": "Review {{ item }}?"})
        ] = Field(alias="decision")
        only: Annotated[Literal["ok"], Choice(description="Fine")] = Field(
            description="Which?"
        )
        yes: Annotated[
            float, Noul("OK?", criteria={"true": ["yes"], "false": {"no": "no"}})
        ]

    def handler(request):
        body = json.loads(request.content)
        assert body["questions"]["action"]["instructions"] == {
            "question": "Review test?"
        }
        assert body["questions"]["action"]["criteria"] == {
            "allow": None,
            "review": None,
        }
        assert body["questions"]["only"]["criteria"] == {"ok": "Fine"}
        assert body["questions"]["yes"]["criteria"] == {
            "true": ["yes"],
            "false": {"no": "no"},
        }
        return httpx.Response(
            200,
            json={
                "answers": {
                    "action": {"type": "choice", "choice": "review"},
                    "only": {"type": "choice", "choice": "ok"},
                    "yes": {"type": "noul", "noul": 0.6},
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = instructor.from_provider(
            "typesafe/jev-latest",
            mode=instructor.Mode.DECISIONS,
            api_key="key",
            http_client=http,
        )
        result = client.create(response_model=Model, context={"item": "test"})
        assert result.action is Action.REVIEW
        assert result.only == "ok"


def test_invalid_models_are_rejected_before_request():
    def handler(_request):
        pytest.fail("invalid model should not make a request")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = instructor.from_provider(
            "typesafe/jev-latest",
            mode=instructor.Mode.DECISIONS,
            api_key="key",
            http_client=http,
        )

        class MissingInstruction(BaseModel):
            category: Literal["a", "b"]

        class InvalidScore(BaseModel):
            score: Annotated[float, Score("How?", levels=["a", "b"], scale=(10, 0))]

        class InvalidExample(BaseModel):
            score: Annotated[
                float, Score("How?", levels=["a", "b"], examples=[("x", 2)])
            ]

        class ConflictingCriteria(BaseModel):
            flag: Annotated[
                float, Noul("OK?", when_true="yes", criteria={"true": "yes"})
            ]

        for model in (
            MissingInstruction,
            InvalidScore,
            InvalidExample,
            ConflictingCriteria,
        ):
            with pytest.raises(ValueError):
                client.create(response_model=model, context={})
        with pytest.raises(TypeError):
            client.create(response_model=MissingInstruction, context=cast(Any, "text"))
        with pytest.raises(ValueError, match="JSON-compatible"):
            client.create(response_model=Decision, context={"post": {1: "changed"}})


@pytest.mark.parametrize(
    "field,answer",
    [
        ("category", {"type": "choice", "choice": "unknown"}),
        ("violation", {"type": "noul", "noul": 1.2}),
        ("severity", {"type": "score", "score": -1}),
        ("target", {"type": "noul", "noul": 0.5}),
        ("format", None),
    ],
)
def test_malformed_answers_fail(field, answer):
    raw = {**RAW, "answers": {**RAW["answers"], field: answer}}
    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=raw))
    ) as http:
        client = instructor.from_provider(
            "openrouter/typesafe/jev-1.13",
            mode=instructor.Mode.DECISIONS,
            api_key="key",
            http_client=http,
        )
        with pytest.raises(ValueError):
            client.create(response_model=Decision, context=CONTEXT)


def test_http_errors_and_environment_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")

    def handler(request):
        assert request.headers["authorization"] == "Bearer env-key"
        return httpx.Response(429, json={"error": "rate limited"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = instructor.from_provider(
            "typesafe/jev-latest", mode=instructor.Mode.DECISIONS, http_client=http
        )
        with pytest.raises(httpx.HTTPStatusError) as exc:
            client.create(response_model=Decision, context=CONTEXT)
        assert exc.value.response.status_code == 429


def test_async_validators_fail_before_request():
    from instructor.v2.validation.async_validators import async_field_validator

    class Model(BaseModel):
        label: Literal["ok"] = Field(description="What is the label?")

        @async_field_validator("label")
        async def validate_label(cls, value: str) -> str:
            return value

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("unexpected request")
        )
    ) as http:
        client = instructor.from_provider(
            "typesafe/jev-latest",
            mode=instructor.Mode.DECISIONS,
            api_key="key",
            http_client=http,
        )
        with pytest.raises(ValueError, match="async validators are not supported"):
            client.create(response_model=Model, context={})


if TYPE_CHECKING:
    from typing_extensions import assert_type

    typed_client = instructor.from_provider(
        "typesafe/jev-latest", mode=instructor.Mode.DECISIONS, api_key="key"
    )
    typed_decision = typed_client.create(response_model=Decision, context=CONTEXT)
    assert_type(typed_decision, Decision)
    assert_type(typed_decision.category, Category)
    assert_type(typed_decision.target, Literal["person", "group"])
