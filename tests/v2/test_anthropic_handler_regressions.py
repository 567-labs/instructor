"""Offline regressions for Anthropic v2 handlers.

Covers native structured outputs via ``output_config``, ``BetaMessage``
responses from ``from_anthropic(beta=True)``, and forced tool_choice names
for models with a custom schema title. Uses real ``anthropic.types`` objects
and a local HTTP transport; nothing under test is mocked.
"""

from __future__ import annotations

import copy
from importlib import import_module
import json
import warnings
from typing import Any

import anthropic
import pytest
from anthropic.types import Message
from anthropic.types.beta import BetaMessage
from pydantic import BaseModel, ConfigDict

import instructor
from instructor import Mode, Provider
from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.registry import mode_registry

# The SDK's HTTP library (httpx or httpx2, depending on the SDK version).
httpx: Any = import_module(
    anthropic.DefaultHttpxClient.__base__.__module__.split(".")[0]
)


class User(BaseModel):
    name: str
    age: int


def _handler(mode: Mode) -> Any:
    handlers = mode_registry.get_handlers(Provider.ANTHROPIC, mode)
    return handlers.request_handler.__self__  # type: ignore[attr-defined]


def _message_payload(
    content: list[dict[str, Any]], stop_reason: str = "end_turn"
) -> dict[str, Any]:
    return {
        "id": "msg_local",
        "type": "message",
        "role": "assistant",
        "model": "claude-local",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _base_kwargs(**extra: Any) -> dict[str, Any]:
    return {
        "model": "claude-local",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Ada is 36"}],
        **extra,
    }


# --- JSON_SCHEMA uses GA output_config ------------------------------------


def test_json_schema_uses_output_config_format() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, result = _handler(Mode.JSON_SCHEMA).prepare_request(User, _base_kwargs())

    assert result["output_config"] == {
        "format": {"type": "json_schema", "schema": anthropic.transform_schema(User)}
    }
    assert "output_format" not in result
    assert "betas" not in result
    assert "tools" not in result
    assert "tool_choice" not in result
    # Native mode must not inject the JSON-mode schema prompt.
    assert "system" not in result


def test_json_schema_preserves_caller_output_config_without_mutation() -> None:
    caller_config = {"effort": "low"}
    kwargs = _base_kwargs(
        output_config=caller_config,
        tools=[{"name": "stale"}],
        tool_choice={"type": "auto"},
    )
    snapshot = copy.deepcopy(kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, result = _handler(Mode.JSON_SCHEMA).prepare_request(User, kwargs)

    assert result["output_config"]["effort"] == "low"
    assert result["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in result
    assert "tool_choice" not in result
    assert kwargs == snapshot
    assert caller_config == {"effort": "low"}


@pytest.mark.parametrize("sentinel", [anthropic.NOT_GIVEN, anthropic.omit, None])
def test_json_schema_treats_sdk_sentinels_as_empty_config(sentinel: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, result = _handler(Mode.JSON_SCHEMA).prepare_request(
            User, _base_kwargs(output_config=sentinel)
        )

    assert set(result["output_config"]) == {"format"}


def _anthropic_client(
    requests: list[dict[str, Any]], response: dict[str, Any]
) -> anthropic.Anthropic:
    def transport(request: Any) -> Any:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response)

    return anthropic.Anthropic(
        api_key="local-only",
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )


@pytest.mark.parametrize("beta", [False, True])
def test_json_schema_round_trip_over_http(beta: bool) -> None:
    requests: list[dict[str, Any]] = []
    sdk = _anthropic_client(
        requests,
        _message_payload([{"type": "text", "text": '{"name":"Ada","age":36}'}]),
    )
    client = instructor.from_anthropic(sdk, mode=Mode.JSON_SCHEMA, beta=beta)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        user = client.create(
            response_model=User,
            max_retries=0,
            output_config={"effort": "low"},
            **_base_kwargs(),
        )

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
    sdk.close()


# --- BetaMessage responses -------------------------------------------------


def test_json_mode_parses_beta_message() -> None:
    response = BetaMessage.model_validate(
        _message_payload([{"type": "text", "text": '{"name":"Ada","age":36}'}])
    )
    result = _handler(Mode.JSON).parse_response(response, User)
    assert result.model_dump() == {"name": "Ada", "age": 36}


def test_json_schema_mode_parses_beta_message() -> None:
    response = BetaMessage.model_validate(
        _message_payload([{"type": "text", "text": '{"name":"Ada","age":36}'}])
    )
    result = _handler(Mode.JSON_SCHEMA).parse_response(response, User)
    assert result.model_dump() == {"name": "Ada", "age": 36}


@pytest.mark.parametrize("message_cls", [Message, BetaMessage])
def test_tools_mode_rejects_max_tokens_for_message_types(message_cls: Any) -> None:
    response = message_cls.model_validate(
        _message_payload(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_local",
                    "name": "User",
                    "input": {"name": "Ada", "age": 36},
                }
            ],
            stop_reason="max_tokens",
        )
    )
    with pytest.raises(IncompleteOutputException):
        _handler(Mode.TOOLS).parse_response(response, User)


# --- Forced tool_choice uses the declared tool name ------------------------


class TitledUser(BaseModel):
    model_config = ConfigDict(title="extract_user")

    name: str


def test_forced_tool_choice_matches_schema_title() -> None:
    _, result = _handler(Mode.TOOLS).prepare_request(TitledUser, _base_kwargs())

    assert result["tools"][0]["name"] == "extract_user"
    assert result["tool_choice"]["type"] == "tool"
    assert result["tool_choice"]["name"] == "extract_user"
