"""Parameterized handler tests for all v2 providers.

These tests exercise handler methods (prepare_request, parse_response, handle_reask)
with shared scenarios and provider-specific mock responses.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from instructor import Mode, Provider
from instructor.processing.function_calls import ResponseSchema
from instructor.v2.core.registry import mode_registry
from tests.v2.provider_matrix import PROVIDER_HANDLER_MODES

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HANDLER_MODULE_PATHS: dict[Provider, Path] = {
    Provider.OPENAI: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.ANYSCALE: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.TOGETHER: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.DATABRICKS: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.DEEPSEEK: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.ANTHROPIC: _PROJECT_ROOT / "instructor/v2/providers/anthropic/handlers.py",
    Provider.GENAI: _PROJECT_ROOT / "instructor/v2/providers/genai/handlers.py",
    Provider.GEMINI: _PROJECT_ROOT / "instructor/v2/providers/gemini/handlers.py",
    Provider.VERTEXAI: _PROJECT_ROOT / "instructor/v2/providers/vertexai/handlers.py",
    Provider.COHERE: _PROJECT_ROOT / "instructor/v2/providers/cohere/handlers.py",
    Provider.PERPLEXITY: _PROJECT_ROOT
    / "instructor/v2/providers/perplexity/handlers.py",
    Provider.XAI: _PROJECT_ROOT / "instructor/v2/providers/xai/handlers.py",
    Provider.GROQ: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.MISTRAL: _PROJECT_ROOT / "instructor/v2/providers/mistral/handlers.py",
    Provider.FIREWORKS: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.BEDROCK: _PROJECT_ROOT / "instructor/v2/providers/bedrock/handlers.py",
    Provider.CEREBRAS: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.WRITER: _PROJECT_ROOT / "instructor/v2/providers/writer/handlers.py",
    Provider.OPENROUTER: _PROJECT_ROOT
    / "instructor/v2/providers/openrouter/handlers.py",
    Provider.REQUESTY: _PROJECT_ROOT / "instructor/v2/providers/requesty/handlers.py",
}
_HANDLERS_LOADED: set[Provider] = set()


def _ensure_handlers_loaded(provider: Provider) -> None:
    if provider in _HANDLERS_LOADED:
        return
    provider_modes = PROVIDER_HANDLER_MODES.get(provider, [])
    if provider_modes and all(
        mode_registry.is_registered(provider, mode) for mode in provider_modes
    ):
        _HANDLERS_LOADED.add(provider)
        return
    handler_path = _HANDLER_MODULE_PATHS.get(provider)
    if handler_path is None:
        return
    spec = importlib.util.spec_from_file_location(
        f"tests.v2.handlers_{provider.value}",
        handler_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load handler module for {provider}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _HANDLERS_LOADED.add(provider)


def _get_handlers(provider: Provider, mode: Mode):
    _ensure_handlers_loaded(provider)
    return mode_registry.get_handlers(provider, mode)


class Answer(ResponseSchema):
    """Simple answer model for handler tests."""

    answer: float


PARSE_SCENARIOS: dict[Provider, dict[Mode, str]] = {
    Provider.OPENAI: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
        Mode.RESPONSES_TOOLS: "responses_output",
    },
    Provider.ANYSCALE: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.TOGETHER: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.DATABRICKS: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.DEEPSEEK: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.OPENROUTER: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.REQUESTY: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.COHERE: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.XAI: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.PERPLEXITY: {
        Mode.MD_JSON: "markdown",
    },
    Provider.GENAI: {
        Mode.JSON: "text",
    },
    Provider.GEMINI: {
        Mode.TOOLS: "tool_call",
        Mode.MD_JSON: "markdown",
    },
    Provider.GROQ: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.MISTRAL: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.FIREWORKS: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.BEDROCK: {
        Mode.TOOLS: "tool_call",
        Mode.TOOLS_STRICT: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.CEREBRAS: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.WRITER: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.VERTEXAI: {
        Mode.TOOLS: "tool_call",
        Mode.MD_JSON: "text",
    },
}


def _dependency_missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except ModuleNotFoundError:
        return True


def _skip_if_missing(module: str) -> None:
    if _dependency_missing(module):
        pytest.skip(
            f"{module} is not installed"  # ty: ignore[too-many-positional-arguments]
        )


def _provider_mode_params():
    params = []
    for provider, modes in PROVIDER_HANDLER_MODES.items():
        for mode in modes:
            params.append(
                pytest.param(provider, mode, id=f"{provider.value}-{mode.value}")
            )
    return params


@dataclass(frozen=True)
class MockResponseBuilder:
    """Builds provider-specific mock responses."""

    provider: Provider

    def tool_response(self, args: dict[str, Any]) -> Any:
        if self.provider == Provider.OPENAI:
            tool_call = SimpleNamespace(
                function=SimpleNamespace(
                    name="Answer",
                    arguments=json.dumps(args),
                )
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        if self.provider == Provider.COHERE:
            tool_call = SimpleNamespace(parameters=args)
            return SimpleNamespace(tool_calls=[tool_call])
        if self.provider == Provider.XAI:
            tool_call = SimpleNamespace(
                function=SimpleNamespace(arguments=json.dumps(args))
            )
            return SimpleNamespace(tool_calls=[tool_call])
        if self.provider == Provider.BEDROCK:
            return {
                "output": {
                    "message": {
                        "content": [
                            {
                                "toolUse": {
                                    "name": "Answer",
                                    "input": args,
                                }
                            }
                        ]
                    }
                }
            }
        if self.provider in {Provider.GEMINI, Provider.VERTEXAI}:
            function_call = SimpleNamespace(name="Answer", args=args)
            part = SimpleNamespace(function_call=function_call)
            content = SimpleNamespace(parts=[part])
            candidate = SimpleNamespace(content=content)
            return SimpleNamespace(candidates=[candidate])
        # Groq, Fireworks, Cerebras, and Writer use OpenAI-compatible format
        if self.provider in {
            Provider.GROQ,
            Provider.FIREWORKS,
            Provider.ANYSCALE,
            Provider.TOGETHER,
            Provider.DATABRICKS,
            Provider.DEEPSEEK,
            Provider.OPENROUTER,
            Provider.REQUESTY,
            Provider.PERPLEXITY,
            Provider.CEREBRAS,
            Provider.WRITER,
        }:
            tool_call = SimpleNamespace(
                function=SimpleNamespace(
                    name="Answer",
                    arguments=json.dumps(args),
                )
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        # Mistral uses OpenAI-compatible format but with different structure
        if self.provider == Provider.MISTRAL:
            tool_call = SimpleNamespace(
                function=SimpleNamespace(
                    name="Answer",
                    arguments=json.dumps(args),
                )
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        raise NotImplementedError(f"Tool response not supported for {self.provider}")

    def text_response(self, text: str) -> Any:
        if self.provider == Provider.OPENAI:
            message = SimpleNamespace(content=text, tool_calls=[])
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        if self.provider in {
            Provider.COHERE,
            Provider.XAI,
            Provider.GENAI,
            Provider.GEMINI,
            Provider.VERTEXAI,
        }:
            return SimpleNamespace(text=text)
        if self.provider == Provider.BEDROCK:
            return {
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": text,
                            }
                        ]
                    }
                }
            }
        # Groq, Fireworks, Mistral, Cerebras, and Writer use OpenAI-compatible format
        if self.provider in {
            Provider.GROQ,
            Provider.FIREWORKS,
            Provider.MISTRAL,
            Provider.ANYSCALE,
            Provider.TOGETHER,
            Provider.DATABRICKS,
            Provider.DEEPSEEK,
            Provider.OPENROUTER,
            Provider.REQUESTY,
            Provider.PERPLEXITY,
            Provider.CEREBRAS,
            Provider.WRITER,
        }:
            message = SimpleNamespace(content=text, tool_calls=[])
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        raise NotImplementedError(f"Text response not supported for {self.provider}")

    def markdown_response(self, text: str) -> Any:
        return self.text_response(f"```json\n{text}\n```")

    def responses_output_response(self, args: dict[str, Any]) -> Any:
        if self.provider != Provider.OPENAI:
            raise NotImplementedError("Responses output only applies to OpenAI")
        item = SimpleNamespace(type="function_call", arguments=json.dumps(args))
        return SimpleNamespace(output=[item])

    def reask_response(self) -> Any:
        if self.provider == Provider.ANTHROPIC:
            return SimpleNamespace(
                content=[_AnthropicContent(type="text", text="Invalid response")]
            )
        if self.provider == Provider.GENAI:
            function_call = SimpleNamespace(name="Answer", args={"answer": "invalid"})
            part = SimpleNamespace(function_call=function_call)
            content = SimpleNamespace(parts=[part])
            candidate = SimpleNamespace(content=content)
            return SimpleNamespace(candidates=[candidate])
        if self.provider == Provider.GEMINI:
            function_call = SimpleNamespace(name="Answer", args={"answer": "invalid"})
            part = SimpleNamespace(function_call=function_call)
            return SimpleNamespace(parts=[part], text="Invalid response")
        if self.provider == Provider.VERTEXAI:
            function_call = SimpleNamespace(name="Answer", args={"answer": "invalid"})
            part = SimpleNamespace(function_call=function_call)
            content = SimpleNamespace(parts=[part])
            candidate = SimpleNamespace(content=content)
            return SimpleNamespace(candidates=[candidate], text="Invalid response")
        # Mistral expects OpenAI-compatible format with choices
        # For reask tests, we create a simple message without tool_calls
        # to avoid issues with dump_message expecting Pydantic models
        if self.provider == Provider.MISTRAL:
            # Create a mock that works with dump_message
            # dump_message expects a ChatCompletionMessage-like object
            class MistralMockMessage:
                def __init__(self):
                    self.role = "assistant"
                    self.content = "Invalid response"
                    self.tool_calls = []

                def model_dump(self):
                    return {
                        "role": self.role,
                        "content": self.content,
                        "tool_calls": self.tool_calls,
                    }

            message = MistralMockMessage()
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        if self.provider == Provider.WRITER:

            class WriterMockMessage:
                def __init__(self):
                    self.role = "assistant"
                    self.content = "Invalid response"
                    self.tool_calls = []

                def model_dump(self):
                    return {
                        "role": self.role,
                        "content": self.content,
                        "tool_calls": self.tool_calls,
                    }

            message = WriterMockMessage()
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        if self.provider == Provider.PERPLEXITY:

            class PerplexityMockMessage:
                def __init__(self):
                    self.role = "assistant"
                    self.content = "Invalid response"
                    self.tool_calls = []

                def model_dump(self):
                    return {
                        "role": self.role,
                        "content": self.content,
                        "tool_calls": self.tool_calls,
                    }

            message = PerplexityMockMessage()
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        if self.provider == Provider.BEDROCK:
            return {
                "output": {
                    "message": {
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tool-use-1",
                                    "name": "Answer",
                                    "input": {"answer": "invalid"},
                                }
                            }
                        ]
                    }
                }
            }
        return SimpleNamespace(text="Invalid response")


class _AnthropicContent:
    def __init__(self, type: str, text: str | None = None, id: str | None = None):
        self.type = type
        self.text = text
        self.id = id

    def model_dump(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text, "id": self.id}


@pytest.mark.parametrize("provider,mode", _provider_mode_params())
def test_prepare_request_with_none_model(provider: Provider, mode: Mode) -> None:
    """prepare_request should handle None response_model."""
    if provider == Provider.GENAI:
        _skip_if_missing("google.genai")
    if provider == Provider.GEMINI:
        _skip_if_missing("google.genai")
        _skip_if_missing("google.generativeai")
    if provider == Provider.VERTEXAI:
        _skip_if_missing("vertexai")
    if provider == Provider.OPENAI and mode == Mode.RESPONSES_TOOLS:
        _skip_if_missing("openai")
    if provider == Provider.MISTRAL and mode == Mode.JSON_SCHEMA:
        _skip_if_missing("mistralai")
    if mode == Mode.PARALLEL_TOOLS:
        pytest.skip(
            "Parallel tools requires special response_model setup"  # ty: ignore[too-many-positional-arguments]
        )
    # Anthropic JSON_SCHEMA requires a response_model
    if provider == Provider.ANTHROPIC and mode == Mode.JSON_SCHEMA:
        pytest.skip(
            "Anthropic JSON_SCHEMA mode requires a response_model"  # ty: ignore[too-many-positional-arguments]
        )

    handlers = _get_handlers(provider, mode)
    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result_model, result_kwargs = handlers.request_handler(None, kwargs)

    assert result_model is None
    assert isinstance(result_kwargs, dict)


@pytest.mark.parametrize("provider,mode", _provider_mode_params())
def test_prepare_request_with_model(provider: Provider, mode: Mode) -> None:
    """prepare_request should return a model and kwargs when response_model is set."""
    if provider == Provider.GENAI:
        _skip_if_missing("google.genai")
    if provider == Provider.GEMINI:
        _skip_if_missing("google.genai")
        _skip_if_missing("google.generativeai")
    if provider == Provider.VERTEXAI:
        _skip_if_missing("vertexai")
    if provider == Provider.OPENAI and mode == Mode.RESPONSES_TOOLS:
        _skip_if_missing("openai")
    if provider == Provider.MISTRAL and mode == Mode.JSON_SCHEMA:
        _skip_if_missing("mistralai")
    if mode == Mode.PARALLEL_TOOLS:
        pytest.skip(
            "Parallel tools requires special response_model setup"  # ty: ignore[too-many-positional-arguments]
        )

    handlers = _get_handlers(provider, mode)
    kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
    result_model, result_kwargs = handlers.request_handler(Answer, kwargs)

    assert result_model is not None
    assert isinstance(result_kwargs, dict)


@pytest.mark.parametrize("provider,mode", _provider_mode_params())
def test_parse_response(provider: Provider, mode: Mode) -> None:
    """parse_response should return a validated model for supported scenarios."""
    scenario = PARSE_SCENARIOS.get(provider, {}).get(mode)
    if scenario is None:
        pytest.skip(
            "No parse_response scenario defined for this provider/mode"  # ty: ignore[too-many-positional-arguments]
        )

    handlers = _get_handlers(provider, mode)
    builder = MockResponseBuilder(provider)
    payload = {"answer": 4.0}

    if scenario == "tool_call":
        response = builder.tool_response(payload)
    elif scenario == "text":
        response = builder.text_response(json.dumps(payload))
    elif scenario == "markdown":
        response = builder.markdown_response(json.dumps(payload))
    elif scenario == "responses_output":
        response = builder.responses_output_response(payload)
    else:
        raise ValueError(f"Unsupported scenario {scenario}")

    result = handlers.response_parser(
        response=response,
        response_model=Answer,
        validation_context=None,
        strict=None,
        stream=False,
        is_async=False,
    )

    assert isinstance(result, Answer)
    assert result.answer == 4.0


@pytest.mark.parametrize("provider,mode", _provider_mode_params())
def test_parse_response_validation_error(provider: Provider, mode: Mode) -> None:
    """parse_response should raise ValidationError on invalid payloads."""
    scenario = PARSE_SCENARIOS.get(provider, {}).get(mode)
    if scenario is None:
        pytest.skip(
            "No parse_response scenario defined for this provider/mode"  # ty: ignore[too-many-positional-arguments]
        )

    handlers = _get_handlers(provider, mode)
    builder = MockResponseBuilder(provider)
    invalid_payload = {"wrong": "field"}

    if scenario == "tool_call":
        response = builder.tool_response(invalid_payload)
    elif scenario == "text":
        response = builder.text_response(json.dumps(invalid_payload))
    elif scenario == "markdown":
        response = builder.markdown_response(json.dumps(invalid_payload))
    elif scenario == "responses_output":
        response = builder.responses_output_response(invalid_payload)
    else:
        raise ValueError(f"Unsupported scenario {scenario}")

    with pytest.raises(ValidationError):
        handlers.response_parser(
            response=response,
            response_model=Answer,
            validation_context=None,
            strict=None,
            stream=False,
            is_async=False,
        )


@pytest.mark.parametrize("provider,mode", _provider_mode_params())
def test_handle_reask_adds_message(provider: Provider, mode: Mode) -> None:
    """handle_reask should return kwargs with messages."""
    if provider == Provider.GENAI:
        _skip_if_missing("google.genai")
    if provider == Provider.GEMINI:
        _skip_if_missing("google.genai")
        _skip_if_missing("google.generativeai")
        _skip_if_missing("google.ai.generativelanguage")
    if provider == Provider.VERTEXAI:
        _skip_if_missing("vertexai")
    handlers = _get_handlers(provider, mode)
    builder = MockResponseBuilder(provider)
    if provider in {Provider.GENAI, Provider.GEMINI, Provider.VERTEXAI}:
        kwargs = {"contents": []}
        expected_key = "contents"
    else:
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        expected_key = "messages"
    response = builder.reask_response()
    exception = ValueError("Validation failed")

    result = handlers.reask_handler(
        kwargs=kwargs,
        response=response,
        exception=exception,
    )

    assert isinstance(result, dict)
    assert expected_key in result
    assert len(result[expected_key]) >= 1
