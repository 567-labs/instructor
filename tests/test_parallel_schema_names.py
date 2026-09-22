import asyncio
from collections.abc import Iterable
from typing import Any, Union

import pytest
from anthropic.types import Message
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict, create_model

from instructor import Mode
from instructor.dsl.parallel import (
    AnthropicParallelModel,
    ParallelModel,
    VertexAIParallelModel,
    handle_anthropic_parallel_model,
    handle_parallel_model,
)
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.core.response import (
    process_response,
    process_response_async,
)


@pytest.fixture(params=["legacy", "sync", "async", "tools"])
def parse_path(request: pytest.FixtureRequest) -> str:
    return request.param


class DefaultName(BaseModel):
    value: int


class ConfiguredName(BaseModel):
    model_config = ConfigDict(title="extract_value")
    value: int


class SchemaExtraName(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "extra_value"})
    value: int


@pytest.mark.parametrize("model", [DefaultName, ConfiguredName, SchemaExtraName])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_parallel_dispatches_declared_schema_name(
    model: type[BaseModel], provider: str, parse_path: str
) -> None:
    assert parse_tools((model,), provider, parse_path) == [model(value=1)]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_parallel_distinguishes_models_with_the_same_class_name(
    provider: str, parse_path: str
) -> None:
    first = create_model(
        "Entry", __config__=ConfigDict(title="first_entry"), value=(int, ...)
    )
    second = create_model(
        "Entry", __config__=ConfigDict(title="second_entry"), value=(int, ...)
    )
    assert parse_tools((first, second), provider, parse_path) == [
        first(value=1),
        second(value=2),
    ]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_parallel_schema_name_can_match_another_class_name(
    provider: str, parse_path: str
) -> None:
    first = create_model(
        "First", __config__=ConfigDict(title="Second"), value=(int, ...)
    )
    second = create_model(
        "Second", __config__=ConfigDict(title="First"), value=(int, ...)
    )
    assert parse_tools((first, second), provider, parse_path) == [
        first(value=1),
        second(value=2),
    ]


def test_vertexai_parallel_preserves_class_name_dispatch() -> None:
    gm = pytest.importorskip("vertexai.generative_models")
    from instructor.v2.providers.vertexai.handlers import _create_vertexai_tool

    tool = _create_vertexai_tool(Iterable[ConfiguredName])
    name = tool.to_dict()["function_declarations"][0]["name"]
    assert name == "ConfiguredName"
    response = gm.GenerationResponse.from_dict(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"function_call": {"name": name, "args": {"value": 1}}}
                        ],
                    }
                }
            ]
        }
    )
    model = VertexAIParallelModel(Iterable[ConfiguredName])
    assert list(model.from_response(response, Mode.VERTEXAI_PARALLEL_TOOLS)) == [
        ConfiguredName(value=1)
    ]


def parse_tools(
    models: tuple[type[BaseModel], ...], provider: str, parse_path: str
) -> list[BaseModel]:
    response_model = Iterable[Union[models]]
    handlers = mode_registry.get_handlers(Provider(provider), Mode.PARALLEL_TOOLS)
    _, kwargs = handlers.request_handler(
        response_model=response_model,
        kwargs={"messages": [{"role": "user", "content": "Extract the values."}]},
    )
    if provider == "openai":
        schemas = (
            handle_parallel_model(response_model)
            if parse_path == "legacy"
            else kwargs["tools"]
        )
        response = ChatCompletion.model_validate(
            {
                "id": "chatcmpl-parallel",
                "created": 0,
                "model": "gpt-4o-mini",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": f"call_{index}",
                                    "type": "function",
                                    "function": {
                                        "name": schema["function"]["name"],
                                        "arguments": f'{{"value": {index}}}',
                                    },
                                }
                                for index, schema in enumerate(schemas, start=1)
                            ],
                        },
                    }
                ],
            }
        )
        if parse_path == "legacy":
            return list(
                ParallelModel(response_model).from_response(
                    response, Mode.PARALLEL_TOOLS
                )
            )
        return parse_prepared(response, response_model, provider, parse_path)

    schemas = (
        handle_anthropic_parallel_model(response_model)
        if parse_path == "legacy"
        else kwargs["tools"]
    )
    message = Message.model_validate(
        {
            "id": "msg_parallel",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {
                    "id": f"toolu_{index}",
                    "type": "tool_use",
                    "name": schema["name"],
                    "input": {"value": index},
                }
                for index, schema in enumerate(schemas, start=1)
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    if parse_path == "legacy":
        return list(
            AnthropicParallelModel(response_model).from_response(
                message, Mode.ANTHROPIC_PARALLEL_TOOLS
            )
        )
    return parse_prepared(message, response_model, provider, parse_path)


def parse_prepared(
    response: ChatCompletion | Message,
    response_model: Any,
    provider: str,
    parse_path: str,
) -> list[BaseModel]:
    kwargs: dict[str, Any] = {
        "response_model": response_model,
        "mode": Mode.TOOLS if parse_path == "tools" else Mode.PARALLEL_TOOLS,
        "provider": Provider(provider),
        "stream": False,
    }
    if parse_path == "async":
        return list(asyncio.run(process_response_async(response, **kwargs)))
    return list(process_response(response, **kwargs))
