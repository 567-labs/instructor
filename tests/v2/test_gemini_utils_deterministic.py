from __future__ import annotations

import sys
from enum import Enum
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import ConfigurationError
from instructor.v2.providers.gemini import utils
from tests.v2._fake_genai import FakeContent, FakeFile, FakePart, install_fake_genai


class Answer(BaseModel):
    value: int


class FakeGenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeFunctionDeclaration:
    def __init__(self, *, name: str, description: str | None, parameters: Any) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters


class FakeTool:
    def __init__(self, *, function_declarations: list[Any]) -> None:
        self.function_declarations = function_declarations


class FakeFunctionCallingConfigMode(Enum):
    ANY = "ANY"


class FakeFunctionCallingConfig:
    def __init__(self, *, mode: Any, allowed_function_names: list[str]) -> None:
        self.mode = mode
        self.allowed_function_names = allowed_function_names


class FakeToolConfig:
    def __init__(self, *, function_calling_config: Any) -> None:
        self.function_calling_config = function_calling_config


class FakeHarmBlockThreshold(Enum):
    OFF = 0
    BLOCK_ONLY_HIGH = 1


class FakeHarmCategory(Enum):
    HARM_CATEGORY_UNSPECIFIED = "unspecified"
    HARM_CATEGORY_JAILBREAK = "jailbreak"
    HARM_CATEGORY_IMAGE_SAFETY = "image"
    HARM_CATEGORY_HATE_SPEECH = "hate"
    HARM_CATEGORY_HARASSMENT = "harassment"
    HARM_CATEGORY_DANGEROUS_CONTENT = "dangerous"


def _install_fake_genai_types(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_genai(
        monkeypatch,
        extra_types={
            "GenerateContentConfig": FakeGenerateContentConfig,
            "FunctionDeclaration": FakeFunctionDeclaration,
            "Tool": FakeTool,
            "ToolConfig": FakeToolConfig,
            "FunctionCallingConfig": FakeFunctionCallingConfig,
            "FunctionCallingConfigMode": FakeFunctionCallingConfigMode,
            "HarmBlockThreshold": FakeHarmBlockThreshold,
            "HarmCategory": FakeHarmCategory,
        },
    )


def _install_fake_vertexai_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    handlers_module = ModuleType("instructor.v2.providers.vertexai.handlers")
    handlers_module.__dict__.update(
        {
            "vertexai_process_response": lambda _kwargs, _model: (
                ["content"],
                ["tool"],
                "tool-config",
            ),
            "vertexai_process_json_response": lambda _kwargs, _model: (
                ["json-content"],
                "generation-config",
            ),
        }
    )

    parallel_module = ModuleType("instructor.v2.providers.vertexai.parallel")
    parallel_module.__dict__["VertexAIParallelModel"] = lambda typehint: (
        "parallel",
        typehint,
    )

    monkeypatch.setitem(
        sys.modules,
        "instructor.v2.providers.vertexai.handlers",
        handlers_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "instructor.v2.providers.vertexai.parallel",
        parallel_module,
    )


def test_transform_to_gemini_prompt_merges_system_messages() -> None:
    result = utils.transform_to_gemini_prompt(
        [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "and brief"},
        ]
    )

    assert result[0]["role"] == "user"
    assert result[0]["parts"][0] == "*be helpful\n\nand brief*"
    assert result[0]["parts"][1] == "hello"
    assert result[1] == {"role": "model", "parts": ["hi"]}


def test_transform_to_gemini_prompt_preserves_multipart_system_text() -> None:
    result = utils.transform_to_gemini_prompt(
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "be helpful"},
                    {"type": "text", "text": "and brief"},
                ],
            },
            {"role": "user", "content": "hello"},
        ]
    )

    assert result[0]["parts"] == ["*be helpful\n\nand brief*", "hello"]


def test_extract_genai_system_message_validates_edge_cases() -> None:
    assert (
        utils.extract_genai_system_message(
            [
                {"role": "system", "content": ["one", "two"]},
                {"role": "user", "content": "hello"},
            ]
        )
        == "one\n\ntwo\n\n"
    )

    with pytest.raises(ValueError, match="At least one user message"):
        utils.extract_genai_system_message([{"role": "system", "content": "only"}])

    with pytest.raises(ValueError, match="Jinja templating"):
        utils.extract_genai_system_message(
            [
                {"role": "system", "content": "Hello {{ name }}"},
                {"role": "user", "content": "hi"},
            ]
        )


def test_convert_to_genai_messages_supports_strings_existing_content_and_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)

    class FakeImage:
        def to_genai(self) -> str:
            return "image-part"

    image = FakeImage()
    monkeypatch.setattr(utils, "Image", FakeImage)

    existing = FakeContent(role="user", parts=[FakePart.from_text("existing")])
    uploaded = FakeFile()

    messages: list[Any] = [
        "hello",
        existing,
        uploaded,
        {"role": "user", "content": ["one", image]},
    ]
    result = utils.convert_to_genai_messages(messages)

    assert result[0].role == "user"
    assert result[0].parts[0].text == "hello"
    assert result[1] is existing
    assert result[2] is uploaded
    assert result[3].parts[0].text == "one"
    assert result[3].parts[1] == "image-part"


def test_handle_genai_message_conversion_extracts_system_and_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)
    monkeypatch.setattr(
        utils,
        "convert_to_genai_messages",
        lambda messages: ["converted", *messages],
    )
    monkeypatch.setattr(
        "instructor.v2.core.multimodal.extract_genai_multimodal_content",
        lambda contents, autodetect_images: [*contents, autodetect_images],
    )

    result = utils.handle_genai_message_conversion(
        {
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "hello"},
            ]
        },
        autodetect_images=True,
    )

    assert result["contents"][-1] is True
    assert result["config"].kwargs["system_instruction"] == "rules\n\n"
    assert "messages" not in result


def test_handle_genai_message_conversion_folds_generation_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression test for #2366: when response_model is None, generation_config must be folded
    # into `config` instead of leaking as a raw keyword argument to generate_content.
    _install_fake_genai_types(monkeypatch)
    monkeypatch.setattr(
        utils, "convert_to_genai_messages", lambda _messages: ["converted"]
    )
    monkeypatch.setattr(
        "instructor.v2.core.multimodal.extract_genai_multimodal_content",
        lambda contents, _autodetect_images: contents,
    )

    result = utils.handle_genai_message_conversion(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "generation_config": {"temperature": 0.7, "max_tokens": 128},
        }
    )

    assert "generation_config" not in result
    assert result["config"].kwargs["temperature"] == 0.7
    assert result["config"].kwargs["max_output_tokens"] == 128


def test_handle_vertexai_wrappers_use_provider_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vertexai_wrappers(monkeypatch)

    parallel_model, parallel_kwargs = utils.handle_vertexai_parallel_tools(
        tuple[Answer],
        {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert parallel_model == ("parallel", tuple[Answer])
    assert parallel_kwargs["contents"] == ["content"]

    model, tool_kwargs = utils.handle_vertexai_tools(
        Answer,
        {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert model is Answer
    assert tool_kwargs["tools"] == ["tool"]
    assert tool_kwargs["tool_config"] == "tool-config"

    json_model, json_kwargs = utils.handle_vertexai_json(
        Answer,
        {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert json_model is Answer
    assert json_kwargs["generation_config"] == "generation-config"


def test_handle_vertexai_parallel_tools_rejects_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vertexai_wrappers(monkeypatch)
    with pytest.raises(ConfigurationError, match="stream=True is not supported"):
        utils.handle_vertexai_parallel_tools(
            tuple[Answer],
            {"messages": [], "stream": True},
        )


def test_handle_gemini_json_adds_schema_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "_default_safety_thresholds", lambda: None)

    model, kwargs = utils.handle_gemini_json(
        Answer,
        {"messages": [{"role": "user", "content": "hello"}]},
    )

    assert model is Answer
    assert kwargs["contents"][0]["role"] == "user"
    assert "json_schema" in kwargs["contents"][0]["parts"][0]
    assert kwargs["generation_config"]["response_mime_type"] == "application/json"


def test_update_gemini_kwargs_does_not_mutate_generation_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils, "_default_safety_thresholds", lambda: None)
    generation_config = {"max_tokens": 5}

    result = utils.update_gemini_kwargs({"generation_config": generation_config})

    assert generation_config == {"max_tokens": 5}
    assert result["generation_config"] == {"max_output_tokens": 5}


def test_handle_gemini_json_guards_empty_or_missing_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils, "_default_safety_thresholds", lambda: None)

    for kwargs in ({"messages": []}, {}):
        model, result = utils.handle_gemini_json(Answer, kwargs)

        assert model is Answer
        assert result["contents"][0]["role"] == "user"
        assert "json_schema" in result["contents"][0]["parts"][0]
        assert result["generation_config"]["response_mime_type"] == "application/json"


def test_handle_gemini_tools_sets_tool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "_default_safety_thresholds", lambda: None)
    monkeypatch.setattr(Answer, "gemini_schema", {"name": "Answer"}, raising=False)

    model, kwargs = utils.handle_gemini_tools(
        Answer,
        {"messages": [{"role": "user", "content": "hello"}]},
    )

    assert model is Answer
    assert kwargs["tools"] == [{"name": "Answer"}]
    assert kwargs["tool_config"]["function_calling_config"][
        "allowed_function_names"
    ] == ["Answer"]


def test_map_to_gemini_function_schema_raises_helpful_error_when_jsonref_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "jsonref", None)

    with pytest.raises(ConfigurationError, match="instructor\\[google-genai\\]"):
        utils.map_to_gemini_function_schema(Answer.model_json_schema())


def test_update_gemini_kwargs_applies_safety_defaults_without_mutating_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Category:
        def __init__(self, name: str) -> None:
            self.name = name

    hate = Category("HARM_CATEGORY_HATE_SPEECH")
    harassment = Category("HARM_CATEGORY_HARASSMENT")
    monkeypatch.setattr(
        utils,
        "_default_safety_thresholds",
        lambda: {hate: 2, harassment: 1},
    )

    safety_settings = {hate: 3}
    result = utils.update_gemini_kwargs(
        {
            "generation_config": {"max_tokens": 32},
            "messages": [{"role": "user", "content": "hello"}],
            "safety_settings": safety_settings,
        }
    )

    assert safety_settings == {hate: 3}
    assert result["safety_settings"] is not safety_settings
    assert result["generation_config"]["max_output_tokens"] == 32
    assert result["contents"][0]["parts"] == ["hello"]
    assert result["safety_settings"][hate] == 2
    assert result["safety_settings"][harassment] == 1


def test_handle_gemini_json_rejects_model_kwarg() -> None:
    with pytest.raises(ConfigurationError, match="must be set while patching"):
        utils.handle_gemini_json(Answer, {"messages": [], "model": "gemini-pro"})


def test_update_genai_kwargs_merges_generation_safety_and_user_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)

    result = utils.update_genai_kwargs(
        {
            "generation_config": {"max_tokens": 64, "stop": ["done"]},
            "safety_settings": {
                FakeHarmCategory.HARM_CATEGORY_HARASSMENT: FakeHarmBlockThreshold.BLOCK_ONLY_HIGH
            },
            "config": {
                "thinking_config": {"budget": 2},
                "labels": {"team": "dx"},
                "cached_content": "cache-1",
                "automatic_function_calling": True,
            },
        },
        {},
    )

    assert result["max_output_tokens"] == 64
    assert result["stop_sequences"] == ["done"]
    assert result["thinking_config"] == {"budget": 2}
    assert result["labels"] == {"team": "dx"}
    assert result["cached_content"] == "cache-1"
    assert result["automatic_function_calling"] is True
    assert {entry["category"] for entry in result["safety_settings"]} == {
        FakeHarmCategory.HARM_CATEGORY_HATE_SPEECH,
        FakeHarmCategory.HARM_CATEGORY_HARASSMENT,
        FakeHarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    }


def test_handle_genai_structured_outputs_builds_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)
    monkeypatch.setattr(
        utils,
        "convert_to_genai_messages",
        lambda _messages: ["converted"],
    )
    monkeypatch.setattr(
        "instructor.v2.core.multimodal.extract_genai_multimodal_content",
        lambda contents, autodetect_images: [*contents, autodetect_images],
    )
    monkeypatch.setattr(utils, "map_to_gemini_function_schema", lambda schema: schema)
    monkeypatch.setattr(
        utils,
        "update_genai_kwargs",
        lambda _kwargs, base_config: base_config | {"thinking_config": {"budget": 4}},
    )

    model, result = utils.handle_genai_structured_outputs(
        Answer,
        {
            "messages": [{"role": "user", "content": "hello"}],
            "system": "rules",
        },
        autodetect_images=True,
    )

    assert model is Answer
    assert result["contents"] == ["converted", True]
    assert result["config"].kwargs["system_instruction"] == "rules"
    assert result["config"].kwargs["response_schema"] is Answer
    assert result["config"].kwargs["thinking_config"] == {"budget": 4}
    assert "messages" not in result


def test_handle_genai_tools_builds_tool_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)
    monkeypatch.setattr(
        utils,
        "convert_to_genai_messages",
        lambda _messages: ["converted"],
    )
    monkeypatch.setattr(
        "instructor.v2.core.multimodal.extract_genai_multimodal_content",
        lambda contents, autodetect_images: [*contents, autodetect_images],
    )
    monkeypatch.setattr(utils, "map_to_genai_schema", lambda _schema: {"schema": "ok"})
    monkeypatch.setattr(
        utils,
        "update_genai_kwargs",
        lambda _kwargs, base_config: base_config,
    )

    model, result = utils.handle_genai_tools(
        Answer,
        {
            "messages": [{"role": "user", "content": "hello"}],
            "system": "rules",
        },
        autodetect_images=False,
    )

    assert model is Answer
    declaration = result["config"].kwargs["tools"][0].function_declarations[0]
    assert declaration.name == "Answer"
    assert declaration.parameters == {"schema": "ok"}
    assert result["config"].kwargs["tool_config"].function_calling_config.mode is (
        FakeFunctionCallingConfigMode.ANY
    )
    assert result["config"].kwargs["system_instruction"] == "rules"


def test_handle_genai_tools_without_model_uses_message_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)
    expected = {"contents": ["converted"]}
    monkeypatch.setattr(
        utils,
        "handle_genai_message_conversion",
        lambda _kwargs, autodetect_images: expected | {"autodetect": autodetect_images},
    )

    model, result = utils.handle_genai_tools(
        None, {"messages": []}, autodetect_images=True
    )

    assert model is None
    assert result["autodetect"] is True
