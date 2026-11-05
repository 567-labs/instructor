"""Shared handler tests for OpenAI-compatible v2 providers."""

from __future__ import annotations

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.core.exceptions import IncompleteOutputException
from instructor.v2.core.registry import mode_registry, normalize_mode
from tests.coverage._openai import chat_completion, tool_call
from tests.v2.provider_matrix import PROVIDER_SPECS

OPENAI_COMPAT_PROVIDERS = (
    Provider.GROQ,
    Provider.FIREWORKS,
    Provider.CEREBRAS,
    Provider.WRITER,
)
OPENAI_HANDLER_ALIAS_PROVIDERS = (
    Provider.GROQ,
    Provider.FIREWORKS,
    Provider.CEREBRAS,
)


class Answer(BaseModel):
    answer: float


def _handlers(provider: Provider, mode: Mode):
    return mode_registry.get_handlers(provider, mode)


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
def test_tools_request_preserves_kwargs(provider: Provider) -> None:
    kwargs = {
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 500,
        "temperature": 0.7,
    }
    original = kwargs.copy()

    _, result_kwargs = _handlers(provider, Mode.TOOLS).request_handler(Answer, kwargs)

    assert kwargs == original
    assert result_kwargs["max_tokens"] == 500
    assert result_kwargs["temperature"] == 0.7
    assert result_kwargs["tools"][0]["type"] == "function"


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
def test_md_json_extends_existing_system_message(provider: Provider) -> None:
    kwargs = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
    }

    _, result_kwargs = _handlers(provider, Mode.MD_JSON).request_handler(Answer, kwargs)

    system_message = result_kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert "You are helpful." in system_message["content"]
    assert "json_schema" in system_message["content"]


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
@pytest.mark.parametrize(
    ("mode", "response"),
    [
        (
            Mode.TOOLS,
            chat_completion(tool_calls=[tool_call("Answer", {"answer": 5.0})]),
        ),
        (Mode.MD_JSON, chat_completion(content='{"answer": 21.0}')),
    ],
)
def test_response_parser_common_paths(
    provider: Provider,
    mode: Mode,
    response: ChatCompletion,
) -> None:
    result = _handlers(provider, mode).response_parser(
        response,
        Answer,
        validation_context={"test": "context"},
        strict=True,
    )

    assert isinstance(result, Answer)


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
def test_tools_support_nested_models(provider: Provider) -> None:
    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    _, result_kwargs = _handlers(provider, Mode.TOOLS).request_handler(
        Person,
        {"messages": [{"role": "user", "content": "Get person info"}]},
    )

    assert "tools" in result_kwargs


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
def test_incomplete_tools_output_raises(provider: Provider) -> None:
    response = chat_completion(
        tool_calls=[tool_call("Answer", {"answer": 4.0})],
        finish_reason="length",
    )

    with pytest.raises(IncompleteOutputException):
        _handlers(provider, Mode.TOOLS).response_parser(response, Answer)


@pytest.mark.parametrize("provider", OPENAI_HANDLER_ALIAS_PROVIDERS)
@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.MD_JSON])
def test_alias_providers_reuse_openai_handlers(provider: Provider, mode: Mode) -> None:
    provider_handlers = _handlers(provider, mode)
    openai_handlers = _handlers(Provider.OPENAI, mode)

    assert provider_handlers.request_handler == openai_handlers.request_handler
    assert provider_handlers.response_parser == openai_handlers.response_parser


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
def test_legacy_modes_remain_accepted(provider: Provider) -> None:
    spec = PROVIDER_SPECS[provider]

    for legacy_mode in spec.legacy_modes:
        assert normalize_mode(provider, legacy_mode) != legacy_mode
        assert mode_registry.is_registered(provider, legacy_mode)


def test_strict_mode_does_not_pollute_schema_cache() -> None:
    """strict=True must not permanently corrupt the lru_cache for subsequent calls.

    generate_openai_schema uses lru_cache, so it returns the same dict object on
    every call. If the TOOLS handler writes "strict" directly into that dict, all
    future calls for the same model (even without strict=True) receive a schema
    that already carries "strict": True. The fix is a shallow copy before mutation.
    """

    class StrictCacheModel(BaseModel):
        x: int

    handler = _handlers(Provider.OPENAI, Mode.TOOLS)

    # First call with strict=True.
    _, result_strict = handler.request_handler(
        StrictCacheModel,
        {"messages": [{"role": "user", "content": "test"}], "strict": True},
    )
    assert result_strict["tools"][0]["function"].get("strict") is True

    # Second call WITHOUT strict=True must not inherit the mutation.
    _, result_plain = handler.request_handler(
        StrictCacheModel,
        {"messages": [{"role": "user", "content": "test"}]},
    )
    assert "strict" not in result_plain["tools"][0]["function"], (
        "lru_cache was poisoned: 'strict' key survived a non-strict call"
    )


def test_openai_json_schema_mode_emits_strict_compliant_schema() -> None:
    """Mode.JSON_SCHEMA must emit strict: True, additionalProperties: False, and all properties required."""

    class Address(BaseModel):
        street: str
        city: str | None = None

    class User(BaseModel):
        name: str
        address: Address
        tags: list[str] | None = None

    handler = _handlers(Provider.OPENAI, Mode.JSON_SCHEMA)
    _, result = handler.request_handler(
        User,
        {"messages": [{"role": "user", "content": "Extract user"}]},
    )

    rf = result["response_format"]
    assert rf["type"] == "json_schema"
    json_schema = rf["json_schema"]
    assert json_schema["strict"] is True
    assert json_schema["name"] == "User"

    schema = json_schema["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["address", "name", "tags"]
    assert "default" not in schema["properties"]["tags"]

    nested_address = schema["$defs"]["Address"]
    assert nested_address["type"] == "object"
    assert nested_address["additionalProperties"] is False
    assert sorted(nested_address["required"]) == ["city", "street"]
    assert "default" not in nested_address["properties"]["city"]


def test_openai_tools_strict_mode_emits_strict_compliant_schema() -> None:
    """Mode.TOOLS with strict=True must normalize parameters with additionalProperties: False and all required."""

    class Address(BaseModel):
        street: str
        city: str | None = None

    class User(BaseModel):
        name: str
        address: Address
        tags: list[str] | None = None

    handler = _handlers(Provider.OPENAI, Mode.TOOLS)
    _, result = handler.request_handler(
        User,
        {"messages": [{"role": "user", "content": "Extract user"}], "strict": True},
    )

    tool = result["tools"][0]["function"]
    assert tool["strict"] is True

    params = tool["parameters"]
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    assert sorted(params["required"]) == ["address", "name", "tags"]

    nested_address = params["$defs"]["Address"]
    assert nested_address["type"] == "object"
    assert nested_address["additionalProperties"] is False
    assert sorted(nested_address["required"]) == ["city", "street"]
    assert "default" not in nested_address["properties"]["city"]


def test_openrouter_structured_outputs_emits_recursive_strict_schema() -> None:
    """OpenRouter structured outputs must recursively constrain nested $defs and required properties."""
    from instructor.v2.providers.openai.handlers import (
        handle_openrouter_structured_outputs,
    )

    class Address(BaseModel):
        street: str
        city: str | None = None

    class User(BaseModel):
        name: str
        address: Address
        tags: list[str] | None = None

    _, result = handle_openrouter_structured_outputs(
        User,
        {"messages": [{"role": "user", "content": "Extract user"}]},
    )

    rf = result["response_format"]
    assert rf["type"] == "json_schema"
    json_schema = rf["json_schema"]
    assert json_schema["strict"] is True

    schema = json_schema["schema"]
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["address", "name", "tags"]

    nested_address = schema["$defs"]["Address"]
    assert nested_address["type"] == "object"
    assert nested_address["additionalProperties"] is False
    assert sorted(nested_address["required"]) == ["city", "street"]
    assert "default" not in nested_address["properties"]["city"]


def test_openai_json_schema_mode_allows_explicit_strict_false() -> None:
    """Mode.JSON_SCHEMA with strict=False skips strict normalization."""

    class User(BaseModel):
        name: str
        tags: list[str] | None = None

    handler = _handlers(Provider.OPENAI, Mode.JSON_SCHEMA)
    _, result = handler.request_handler(
        User,
        {"messages": [{"role": "user", "content": "Extract user"}], "strict": False},
    )

    json_schema = result["response_format"]["json_schema"]
    assert "strict" not in json_schema
    assert "additionalProperties" not in json_schema["schema"]
    assert json_schema["schema"]["required"] == ["name"]
