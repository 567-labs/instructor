from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from instructor import Mode
from instructor.v2.core.errors import ConfigurationError, ResponseParsingError
from instructor.v2.providers.vertexai.parallel import VertexAIParallelModel


class Weather(BaseModel):
    city: str


class Score(BaseModel):
    value: int


def _install_fake_vertexai(monkeypatch: pytest.MonkeyPatch) -> Any:
    class FakePart:
        def __init__(
            self,
            *,
            text: str | None = None,
            function_call: Any = None,
            function_response: Any = None,
        ) -> None:
            self.text = text
            self.function_call = function_call
            self.function_response = function_response

        @classmethod
        def from_text(cls, text: str) -> FakePart:
            return cls(text=text)

        @classmethod
        def from_function_response(
            cls, *, name: str, response: dict[str, Any]
        ) -> FakePart:
            return cls(function_response={"name": name, "response": response})

    class FakeContent:
        def __init__(self, *, role: str | None = None, parts: list[Any]) -> None:
            self.role = role
            self.parts = parts

    class FakeFunctionDeclaration:
        def __init__(
            self, *, name: str, description: str | None, parameters: Any
        ) -> None:
            self.name = name
            self.description = description
            self.parameters = parameters

    class FakeTool:
        def __init__(self, *, function_declarations: list[Any]) -> None:
            self.function_declarations = function_declarations

    class FakeGenerationConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeFunctionCallingConfig:
        class Mode:
            ANY = "ANY"

        def __init__(self, *, mode: str) -> None:
            self.mode = mode

    class FakeToolConfig:
        FunctionCallingConfig = FakeFunctionCallingConfig

        def __init__(self, *, function_calling_config: Any) -> None:
            self.function_calling_config = function_calling_config

    gm_module = ModuleType("vertexai.generative_models")
    setattr(gm_module, "Part", FakePart)  # noqa: B010
    setattr(gm_module, "Content", FakeContent)  # noqa: B010
    setattr(gm_module, "FunctionDeclaration", FakeFunctionDeclaration)  # noqa: B010
    setattr(gm_module, "Tool", FakeTool)  # noqa: B010
    setattr(gm_module, "GenerationConfig", FakeGenerationConfig)  # noqa: B010
    setattr(gm_module, "GenerationResponse", object)  # noqa: B010

    preview_gm_module = ModuleType("vertexai.preview.generative_models")
    setattr(preview_gm_module, "ToolConfig", FakeToolConfig)  # noqa: B010

    preview_module = ModuleType("vertexai.preview")
    setattr(preview_module, "generative_models", preview_gm_module)  # noqa: B010

    vertexai_module = ModuleType("vertexai")
    setattr(vertexai_module, "generative_models", gm_module)  # noqa: B010
    setattr(vertexai_module, "preview", preview_module)  # noqa: B010

    monkeypatch.setitem(sys.modules, "vertexai", vertexai_module)
    monkeypatch.setitem(sys.modules, "vertexai.generative_models", gm_module)
    monkeypatch.setitem(sys.modules, "vertexai.preview", preview_module)
    monkeypatch.setitem(
        sys.modules, "vertexai.preview.generative_models", preview_gm_module
    )

    sys.modules.pop("instructor.v2.providers.vertexai.handlers", None)
    return importlib.import_module("instructor.v2.providers.vertexai.handlers")


def test_vertexai_parallel_model_validates_registered_calls() -> None:
    model = VertexAIParallelModel(Iterable[Weather | Score])  # type: ignore[arg-type]
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(
                                name="Weather",
                                args={"city": "Paris"},
                            )
                        ),
                        SimpleNamespace(
                            function_call=SimpleNamespace(
                                name="Unknown",
                                args={"ignored": True},
                            )
                        ),
                    ]
                )
            )
        ]
    )

    valid = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(
                                name="Weather",
                                args={"city": "Paris"},
                            )
                        )
                    ]
                )
            )
        ]
    )
    assert list(model.from_response(valid, mode=Mode.VERTEXAI_TOOLS)) == [
        Weather(city="Paris")
    ]
    with pytest.raises(ResponseParsingError, match="Unknown"):
        list(model.from_response(response, mode=Mode.VERTEXAI_TOOLS))


def test_vertexai_parallel_model_skips_empty_candidates() -> None:
    model = VertexAIParallelModel(Iterable[Weather])  # type: ignore[arg-type]
    assert list(model.from_response(None, mode=Mode.VERTEXAI_TOOLS)) == []
    assert (
        list(
            model.from_response(
                SimpleNamespace(candidates=[]), mode=Mode.VERTEXAI_TOOLS
            )
        )
        == []
    )


def test_vertexai_message_parsers_and_reask_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _install_fake_vertexai(monkeypatch)
    gm = handlers._gm()

    single = handlers.vertexai_message_parser({"role": "user", "content": "hello"})
    mixed = handlers.vertexai_message_parser(
        {"role": "user", "content": ["hello", gm.Part.from_text("there")]}
    )

    assert single.parts[0].text == "hello"
    assert mixed.parts[0].text == "hello"
    assert mixed.parts[1].text == "there"

    with pytest.raises(ValueError, match="Unsupported content type in list"):
        handlers.vertexai_message_parser({"role": "user", "content": ["ok", object()]})

    with pytest.raises(ValueError, match="Unsupported message content type"):
        handlers.vertexai_message_parser({"role": "user", "content": 123})

    response = SimpleNamespace(
        text='{"bad": true}',
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(
                                name="Weather", args={"city": "Paris"}
                            )
                        )
                    ]
                )
            )
        ],
    )

    tools_kwargs = handlers.reask_vertexai_tools(
        {"contents": []}, response, ValueError("bad")
    )
    json_kwargs = handlers.reask_vertexai_json(
        {"contents": []}, response, ValueError("bad")
    )

    assert len(tools_kwargs["contents"]) == 2
    assert tools_kwargs["contents"][1].parts[0].function_response["name"] == "Weather"
    assert "Validation Errors found" in json_kwargs["contents"][1].parts[0].text


def test_vertexai_process_helpers_build_tools_and_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _install_fake_vertexai(monkeypatch)

    contents, tools, tool_config = handlers.vertexai_process_response(
        {"messages": [{"role": "user", "content": "hello"}]},
        Weather,
    )
    json_contents, generation_config = handlers.vertexai_process_json_response(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "generation_config": {"temperature": 0.1},
        },
        Weather,
    )

    assert contents[0].parts[0].text == "hello"
    assert tools[0].function_declarations[0].name == "Weather"
    assert tool_config.function_calling_config.mode == "ANY"
    assert json_contents[0].parts[0].text == "hello"
    assert generation_config.kwargs["response_mime_type"] == "application/json"
    assert generation_config.kwargs["temperature"] == 0.1


def test_vertexai_schema_helper_rejects_type_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _install_fake_vertexai(monkeypatch)

    with pytest.raises(TypeError, match="Expected concrete model class"):
        handlers._create_gemini_json_schema(list[Weather])


def test_vertexai_schema_helper_raises_helpful_error_when_jsonref_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _install_fake_vertexai(monkeypatch)
    monkeypatch.setitem(sys.modules, "jsonref", None)

    with pytest.raises(ConfigurationError, match="instructor\\[vertexai\\]"):
        handlers._create_gemini_json_schema(Weather)
