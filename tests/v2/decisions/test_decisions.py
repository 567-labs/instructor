from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast
from urllib.parse import urlsplit

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

PROVIDERS = [
    ("openrouter", "typesafe/jev-1.13", "https://openrouter.ai/api/alpha/decisions"),
    ("typesafe", "jev-latest", "https://api.typesafe.ai/v1/systemone"),
]


@dataclass
class DecisionEndpoint:
    url: str = ""
    response: Any = field(default_factory=lambda: deepcopy(RAW))
    status: int = 200
    calls: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def decision_endpoint() -> Iterator[DecisionEndpoint]:
    endpoint = DecisionEndpoint()

    class Handler(BaseHTTPRequestHandler):
        def parse_request(self) -> bool:
            if not super().parse_request():
                return False
            self.record = {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": None,
            }
            endpoint.calls.append(self.record)
            return True

        def do_POST(self) -> None:
            self.record["body"] = json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            body = json.dumps(endpoint.response).encode()
            self.send_response(endpoint.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ARG002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def assert_request(endpoint: DecisionEndpoint, model: str, url: str) -> None:
    assert len(endpoint.calls) == 1
    request = endpoint.calls[0]
    assert request["method"] == "POST"
    assert request["path"] == urlsplit(url).path
    assert request["authorization"] == "Bearer test-key"
    body = request["body"]
    assert body["model"] == model
    assert body["state"] == CONTEXT
    questions = body["questions"]
    assert set(questions) == {"category", "violation", "severity", "format", "target"}
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


def assert_decision(result: Decision) -> None:
    assert isinstance(result, Decision)
    assert result.category is Category.HARASSMENT
    assert result.violation == 0.9
    assert result.severity == 7
    assert result.format == "text"
    assert result.target == "person"


@pytest.mark.parametrize(
    "provider,model,url",
    PROVIDERS,
)
@pytest.mark.parametrize("with_completion", [False, True], ids=["typed", "raw"])
def test_request_and_typed_response(
    decision_endpoint, provider, model, url, with_completion
):
    with httpx.Client(trust_env=False) as http:
        client = instructor.from_provider(
            f"{provider}/{model}",
            mode=instructor.Mode.DECISIONS,
            api_key="test-key",
            http_client=http,
            endpoint=decision_endpoint.url + urlsplit(url).path,
        )
        if with_completion:
            result, raw = client.create_with_completion(
                response_model=Decision, context=CONTEXT
            )
            assert raw == RAW
        else:
            result = client.create(response_model=Decision, context=CONTEXT)
        assert_decision(result)
        assert_request(decision_endpoint, model, url)
        client.close()
        assert not http.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,model,url", PROVIDERS)
@pytest.mark.parametrize("with_completion", [False, True], ids=["typed", "raw"])
async def test_async_request_and_typed_response(
    decision_endpoint, provider, model, url, with_completion
):
    async with httpx.AsyncClient(trust_env=False) as http:
        client = instructor.from_provider(
            f"{provider}/{model}",
            mode=instructor.Mode.DECISIONS,
            async_client=True,
            api_key="test-key",
            http_client=http,
            endpoint=decision_endpoint.url + urlsplit(url).path,
        )
        if with_completion:
            result, raw = await client.create_with_completion(
                response_model=Decision, context=CONTEXT
            )
            assert raw == RAW
        else:
            result = await client.create(response_model=Decision, context=CONTEXT)
        assert_decision(result)
        assert_request(decision_endpoint, model, url)
        await client.close()
        assert not http.is_closed


@pytest.mark.parametrize("provider,model,url", PROVIDERS)
def test_default_provider_endpoint(provider, model, url):
    with httpx.Client(trust_env=False) as http:
        client = instructor.from_provider(
            f"{provider}/{model}",
            mode=instructor.Mode.DECISIONS,
            api_key="test-key",
            http_client=http,
        )
        assert client.endpoint == url


def test_plain_enum_structured_question_and_alias(decision_endpoint):
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

    decision_endpoint.response = {
        "answers": {
            "action": {"type": "choice", "choice": "review"},
            "only": {"type": "choice", "choice": "ok"},
            "yes": {"type": "noul", "noul": 0.6},
        }
    }

    with httpx.Client(trust_env=False) as http:
        client = instructor.from_provider(
            "typesafe/jev-latest",
            mode=instructor.Mode.DECISIONS,
            api_key="key",
            http_client=http,
            endpoint=decision_endpoint.url,
        )
        result = client.create(response_model=Model, context={"item": "test"})
        assert result.action is Action.REVIEW
        assert result.only == "ok"
        assert result.yes == 0.6
        assert result.model_dump(by_alias=True)["decision"] is Action.REVIEW
    assert len(decision_endpoint.calls) == 1
    questions = decision_endpoint.calls[0]["body"]["questions"]
    assert questions["action"]["instructions"] == {"question": "Review test?"}
    assert questions["action"]["criteria"] == {"allow": None, "review": None}
    assert questions["only"]["criteria"] == {"ok": "Fine"}
    assert questions["yes"]["criteria"] == {
        "true": ["yes"],
        "false": {"no": "no"},
    }


def test_invalid_models_are_rejected_before_request(decision_endpoint):
    with httpx.Client(trust_env=False) as http:
        client = instructor.from_provider(
            "typesafe/jev-latest",
            mode=instructor.Mode.DECISIONS,
            api_key="key",
            http_client=http,
            endpoint=decision_endpoint.url,
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

        for model, error in (
            (
                MissingInstruction,
                "Instructions and criteria must be strings, objects, or arrays",
            ),
            (InvalidScore, "score: scale bounds must be finite and increasing"),
            (InvalidExample, "score: invalid Score example level index"),
            (
                ConflictingCriteria,
                "flag: criteria cannot be combined with when_true or when_false",
            ),
        ):
            with pytest.raises(ValueError, match=error):
                client.create(response_model=model, context={})
        with pytest.raises(TypeError, match="context must be a dictionary"):
            client.create(response_model=MissingInstruction, context=cast(Any, "text"))
        with pytest.raises(ValueError, match="JSON-compatible"):
            client.create(response_model=Decision, context={"post": {1: "changed"}})
    assert decision_endpoint.calls == []


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
@pytest.mark.parametrize("async_client", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_malformed_answers_fail(decision_endpoint, field, answer, async_client):
    decision_endpoint.response["answers"][field] = answer
    options: dict[str, Any] = {
        "mode": instructor.Mode.DECISIONS,
        "api_key": "key",
        "endpoint": decision_endpoint.url,
    }
    if async_client:
        async with httpx.AsyncClient(trust_env=False) as http:
            client = instructor.from_provider(
                "openrouter/typesafe/jev-1.13",
                async_client=True,
                http_client=http,
                **options,
            )
            with pytest.raises(ValueError, match=f"{field}:.*answer"):
                await client.create(response_model=Decision, context=CONTEXT)
    else:
        with httpx.Client(trust_env=False) as http:
            client = instructor.from_provider(
                "openrouter/typesafe/jev-1.13", http_client=http, **options
            )
            with pytest.raises(ValueError, match=f"{field}:.*answer"):
                client.create(response_model=Decision, context=CONTEXT)
    assert len(decision_endpoint.calls) == 1


@pytest.mark.parametrize("provider,model,url", PROVIDERS)
@pytest.mark.parametrize("async_client", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_http_errors_and_environment_key(
    decision_endpoint, monkeypatch, provider, model, url, async_client
):
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "env-key")
    decision_endpoint.status = 429
    decision_endpoint.response = {"error": "rate limited"}
    options: dict[str, Any] = {
        "mode": instructor.Mode.DECISIONS,
        "endpoint": decision_endpoint.url + urlsplit(url).path,
    }
    if async_client:
        async with httpx.AsyncClient(trust_env=False) as http:
            client = instructor.from_provider(
                f"{provider}/{model}", async_client=True, http_client=http, **options
            )
            with pytest.raises(httpx.HTTPStatusError) as exc:
                await client.create(response_model=Decision, context=CONTEXT)
    else:
        with httpx.Client(trust_env=False) as http:
            client = instructor.from_provider(
                f"{provider}/{model}", http_client=http, **options
            )
            with pytest.raises(httpx.HTTPStatusError) as exc:
                client.create(response_model=Decision, context=CONTEXT)
    assert exc.value.response.status_code == 429
    assert exc.value.response.json() == {"error": "rate limited"}
    assert len(decision_endpoint.calls) == 1
    assert decision_endpoint.calls[0]["authorization"] == "Bearer env-key"


def test_owned_http_client_is_closed_on_exit():
    with instructor.from_provider(
        "typesafe/jev-latest", mode=instructor.Mode.DECISIONS, api_key="key"
    ) as client:
        assert not client._client.is_closed
    assert client._client.is_closed


@pytest.mark.asyncio
async def test_async_owned_http_client_is_closed_on_exit():
    async with instructor.from_provider(
        "typesafe/jev-latest",
        mode=instructor.Mode.DECISIONS,
        async_client=True,
        api_key="key",
    ) as client:
        assert not client._client.is_closed
    assert client._client.is_closed


def test_async_validators_fail_before_request(decision_endpoint):
    from instructor.v2.validation.async_validators import async_field_validator

    class Model(BaseModel):
        label: Literal["ok"] = Field(description="What is the label?")

        @async_field_validator("label")
        async def validate_label(cls, value: str) -> str:
            return value

    with httpx.Client(trust_env=False) as http:
        client = instructor.from_provider(
            "typesafe/jev-latest",
            mode=instructor.Mode.DECISIONS,
            api_key="key",
            http_client=http,
            endpoint=decision_endpoint.url,
        )
        with pytest.raises(ValueError, match="async validators are not supported"):
            client.create(response_model=Model, context={})
    assert decision_endpoint.calls == []


if TYPE_CHECKING:
    from typing_extensions import assert_type

    typed_client = instructor.from_provider(
        "typesafe/jev-latest", mode=instructor.Mode.DECISIONS, api_key="key"
    )
    typed_decision = typed_client.create(response_model=Decision, context=CONTEXT)
    assert_type(typed_decision, Decision)
    assert_type(typed_decision.category, Category)
    assert_type(typed_decision.target, Literal["person", "group"])
