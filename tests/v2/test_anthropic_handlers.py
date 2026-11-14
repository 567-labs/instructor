"""Unit tests for Anthropic v2 handlers.

These tests verify handler behavior without requiring API keys, by calling
the handlers directly with real ``anthropic.types`` objects. One test makes a
local HTTP round trip through ``httpx.MockTransport``.
"""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Iterable
from importlib import import_module
from typing import Any, cast

import anthropic
import pytest
from anthropic.types import Message
from anthropic.types.beta import BetaMessage
from pydantic import BaseModel, ConfigDict

import instructor
from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.core.response import process_response, process_response_async
from instructor.v2.dsl.iterable import IterableModel

# The SDK's HTTP library (httpx or httpx2, depending on the SDK version).
httpx: Any = import_module(
    anthropic.DefaultHttpxClient.__base__.__module__.split(".")[0]
)


class Answer(BaseModel):
    """Simple answer model for testing."""

    answer: float


class TitledAnswer(BaseModel):
    model_config = ConfigDict(title="extract_answer")

    answer: float


class User(BaseModel):
    name: str
    age: int


def _handler(mode: Mode) -> Any:
    handlers = mode_registry.get_handlers(Provider.ANTHROPIC, mode)
    return cast(Any, handlers.request_handler).__self__


def _message_payload(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "msg_local",
        "type": "message",
        "role": "assistant",
        "model": "claude-local",
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


USER_JSON = [{"type": "text", "text": '{"name":"Ada","age":36}'}]
BASE_KWARGS: dict[str, Any] = {
    "model": "claude-local",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Ada is 36"}],
}


@pytest.fixture
def handler():
    """Get the Anthropic tools handler from registry."""
    return mode_registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)


class TestAnthropicToolsHandlerToolChoice:
    """Regression tests for #2477: a forced single-tool tool_choice did not
    set disable_parallel_tool_use, so Anthropic could still emit multiple
    tool_use blocks for the same forced tool, which the single-model parser
    then rejected even though the model's response was otherwise valid."""

    @pytest.mark.parametrize(
        ("model", "tool_name"),
        [(Answer, "Answer"), (TitledAnswer, "extract_answer")],
    )
    def test_single_response_model_disables_parallel_tool_use(
        self, handler, model: type[BaseModel], tool_name: str
    ):
        kwargs = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 100,
        }

        _, result_kwargs = handler.request_handler(model, kwargs)

        # The forced name must match the declared tool (schema title), not
        # the class __name__.
        assert result_kwargs["tools"][0]["name"] == tool_name
        assert result_kwargs["tool_choice"] == {
            "type": "tool",
            "name": tool_name,
            "disable_parallel_tool_use": True,
        }

    def test_parallel_tools_are_not_forced_to_disable_parallel_use(self, handler):
        # Iterable[T] is the parallel-tools case: the model is *expected* to
        # emit multiple tool_use blocks here, so tool_choice must stay "auto"
        # and must not gain disable_parallel_tool_use.
        kwargs = {
            "messages": [{"role": "user", "content": "List some answers"}],
            "max_tokens": 100,
        }

        _, result_kwargs = handler.request_handler(Iterable[Answer], kwargs)

        assert result_kwargs["tool_choice"] == {"type": "auto"}

    def test_explicit_tool_choice_is_not_overridden(self, handler):
        kwargs = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 100,
            "tool_choice": {"type": "auto"},
        }

        _, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_kwargs["tool_choice"] == {"type": "auto"}


@pytest.mark.parametrize(
    ("extra", "expected_extra_config"),
    [
        ({}, {}),
        (
            {
                "output_config": {"effort": "low"},
                "tools": [{"name": "stale"}],
                "tool_choice": {"type": "auto"},
            },
            {"effort": "low"},
        ),
        ({"output_config": anthropic.NOT_GIVEN}, {}),
        ({"output_config": anthropic.omit}, {}),
        ({"output_config": None}, {}),
    ],
    ids=["no-config", "caller-config", "not-given", "omit", "none"],
)
def test_json_schema_uses_output_config_format(
    extra: dict[str, Any], expected_extra_config: dict[str, Any]
) -> None:
    kwargs = {**BASE_KWARGS, **extra}
    # Keep SDK sentinels identical in the snapshot; copies would compare unequal.
    sentinels = (anthropic.NOT_GIVEN, anthropic.omit)
    snapshot = copy.deepcopy(kwargs, {id(s): s for s in sentinels})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, result = _handler(Mode.JSON_SCHEMA).prepare_request(User, kwargs)

    assert result["output_config"] == {
        **expected_extra_config,
        "format": {"type": "json_schema", "schema": anthropic.transform_schema(User)},
    }
    # GA output_config replaces the beta output_format and stale tool kwargs;
    # native mode must not inject the JSON-mode schema prompt.
    for key in ("output_format", "betas", "tools", "tool_choice", "system"):
        assert key not in result
    assert kwargs == snapshot


@pytest.mark.parametrize("beta", [False, True])
def test_json_schema_round_trip_over_http(beta: bool) -> None:
    requests: list[dict[str, Any]] = []

    def transport(request: Any) -> Any:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_message_payload(USER_JSON))

    sdk = anthropic.Anthropic(
        api_key="local-only",
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    client = instructor.from_anthropic(sdk, mode=Mode.JSON_SCHEMA, beta=beta)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        user = client.create(
            response_model=User,
            max_retries=0,
            output_config={"effort": "low"},
            **BASE_KWARGS,
        )
    sdk.close()

    assert user.model_dump() == {"name": "Ada", "age": 36}
    assert len(requests) == 1
    body = requests[0]
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["format"]["schema"]["properties"].keys() == {
        "name",
        "age",
    }
    assert "output_format" not in body
    assert "tools" not in body


@pytest.mark.parametrize("mode", [Mode.JSON, Mode.JSON_SCHEMA])
def test_json_modes_parse_beta_message(mode: Mode) -> None:
    response = BetaMessage.model_validate(_message_payload(USER_JSON))

    result = _handler(mode).parse_response(response, User)

    assert result.model_dump() == {"name": "Ada", "age": 36}


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_process_response_iterable_uses_anthropic_parser(is_async: bool) -> None:
    class U(BaseModel):
        name: str

    response = Message.model_validate(
        {
            **_message_payload(
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_local",
                        "name": "IterableU",
                        "input": {"tasks": [{"name": "a"}]},
                    }
                ]
            ),
            "stop_reason": "tool_use",
        }
    )
    options: dict[str, Any] = {
        "response_model": IterableModel(U),
        "stream": False,
        "mode": Mode.TOOLS,
        "provider": Provider.ANTHROPIC,
    }

    if is_async:
        result: Any = await process_response_async(cast(Any, response), **options)
    else:
        result = process_response(response, **options)

    assert [item.model_dump() for item in result] == [{"name": "a"}]
    assert isinstance(result[0], U)
    assert result._raw_response is response
