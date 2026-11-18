from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from instructor import Mode
from instructor.v2.providers.genai.handlers import (
    GenAIToolsHandler,
    GenAIStructuredOutputsHandler,
    reask_genai_structured_outputs,
    reask_genai_tools,
)
from instructor.v2.providers.genai.request import update_genai_kwargs
from tests.v2._fake_genai import FakeContent, FakePart, install_fake_genai


class FakeGenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install_fake_genai_types(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_genai(
        monkeypatch,
        extra_types={
            "GenerateContentConfig": FakeGenerateContentConfig,
        },
    )


def test_reask_genai_tools_without_function_call_appends_user_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)
    existing_content = FakeContent(role="model", parts=[FakePart(text="hi")])
    response = SimpleNamespace(candidates=[SimpleNamespace(content=existing_content)])

    result = reask_genai_tools({"contents": []}, response, ValueError("bad schema"))

    assert result["contents"][0] is existing_content
    assert result["contents"][1].role == "user"
    assert "Validation Error found" in result["contents"][1].parts[0].text


def test_reask_genai_tools_with_function_call_appends_user_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)
    function_call = SimpleNamespace(name="Answer")
    content = FakeContent(role="model", parts=[FakePart(function_call=function_call)])
    response = SimpleNamespace(candidates=[SimpleNamespace(content=content)])

    result = reask_genai_tools({"contents": []}, response, ValueError("bad schema"))

    assert result["contents"][0] is content
    assert result["contents"][1].role == "user"
    assert result["contents"][1].role != "tool"
    function_response = result["contents"][1].parts[0].function_response
    assert function_response["name"] == "Answer"
    assert "Validation Error found" in function_response["response"]["error"]


@pytest.mark.parametrize(
    "response_kind",
    ["content", "none", "empty", "no_content", "no_parts", "empty_parts"],
)
def test_reask_genai_structured_outputs_appends_user_error(
    response_kind: str,
) -> None:
    types = pytest.importorskip("google.genai.types")
    original_content = types.UserContent(parts=[types.Part.from_text(text="hello")])
    kwargs = {"contents": [original_content]}
    model_content = types.ModelContent(
        parts=[types.Part(text='{"bad": true}', thought_signature=b"signature")]
    )
    response = {
        "content": types.GenerateContentResponse(
            candidates=[types.Candidate(content=model_content)]
        ),
        "none": None,
        "empty": types.GenerateContentResponse(candidates=[]),
        "no_content": types.GenerateContentResponse(candidates=[types.Candidate()]),
        "no_parts": types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model"))]
        ),
        "empty_parts": types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[]))]
        ),
    }[response_kind]

    result = reask_genai_structured_outputs(kwargs, response, ValueError("bad json"))

    assert result["contents"][-1].role == "user"
    assert "bad json" in result["contents"][-1].parts[0].text
    assert result is not kwargs
    assert kwargs["contents"] == [original_content]
    assert result["contents"] is not kwargs["contents"]
    assert result["contents"][0] is original_content
    if response_kind == "content":
        assert [content.role for content in result["contents"]] == [
            "user",
            "model",
            "user",
        ]
        assert result["contents"][-2] is model_content
        assert result["contents"][-2].parts[0].thought_signature == b"signature"
    else:
        assert len(result["contents"]) == 2

    retried = GenAIStructuredOutputsHandler(mode=Mode.JSON).handle_reask(
        result, response, ValueError("still invalid")
    )
    assert retried["contents"] is not result["contents"]
    assert retried["contents"][-1].role == "user"
    assert "bad json" in result["contents"][-1].parts[0].text
    assert "still invalid" in retried["contents"][-1].parts[0].text


def test_tools_handler_prepare_request_without_response_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)
    monkeypatch.setattr(
        "instructor.v2.providers.gemini.utils.convert_to_genai_messages",
        lambda messages: ["converted", *messages],
    )
    monkeypatch.setattr(
        "instructor.v2.providers.genai.handlers.extract_genai_multimodal_content",
        lambda contents, autodetect_images: [*contents, autodetect_images],
    )

    handler = GenAIToolsHandler(mode=Mode.TOOLS)
    model, kwargs = handler.prepare_request(
        None,
        {
            "messages": [{"role": "user", "content": "hello"}],
            "system": "system note",
            "autodetect_images": True,
            "temperature": 0.2,
        },
    )

    assert model is None
    assert kwargs["contents"] == [
        "converted",
        {"role": "user", "content": "hello"},
        True,
    ]
    assert kwargs["config"].kwargs["system_instruction"] == "system note"
    assert "temperature" not in kwargs


def test_structured_outputs_parse_response_unwraps_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = GenAIStructuredOutputsHandler(mode=Mode.JSON)

    class FakeAdapterBase:
        pass

    class FakeAdapter(FakeAdapterBase):
        def __init__(self, content: str) -> None:
            self.content = content

    monkeypatch.setattr(
        "instructor.v2.providers.genai.handlers.AdapterBase",
        FakeAdapterBase,
    )
    monkeypatch.setattr(
        "instructor.v2.providers.genai.handlers.parse_genai_structured_outputs",
        lambda *_args, **_kwargs: FakeAdapter("done"),
    )

    class FakeResponseModel(BaseModel):
        pass

    result = handler.parse_response(
        response=SimpleNamespace(),
        response_model=FakeResponseModel,
        stream=False,
    )

    assert result == "done"


def _raw_request_case(name: str, types: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (extra request kwargs, expected config fields) for a raw request."""
    harassment = types.HarmCategory.HARM_CATEGORY_HARASSMENT
    only_high = types.HarmBlockThreshold.BLOCK_ONLY_HIGH
    safety = [types.SafetySetting(category=harassment, threshold=only_high)]
    return {
        "openai_options": (
            {
                "max_tokens": 16,
                "temperature": 0,
                "top_p": 0.5,
                "n": 1,
                "stop": ["END"],
                "seed": 3,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
            },
            {
                "max_output_tokens": 16,
                "temperature": 0,
                "top_p": 0.5,
                "candidate_count": 1,
                "stop_sequences": ["END"],
                "seed": 3,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
                # No default safety settings or schema on the raw path.
                "safety_settings": None,
                "tools": None,
                "response_schema": None,
            },
        ),
        # Top-level options beat generation_config; explicit config beats both.
        "precedence": (
            {
                "generation_config": {"max_tokens": 8, "temperature": 0.3},
                "temperature": 0.9,
                "config": types.GenerateContentConfig(max_output_tokens=32),
            },
            {"max_output_tokens": 32, "temperature": 0.9},
        ),
        "thinking_and_safety_list": (
            {
                "thinking_config": types.ThinkingConfig(thinking_budget=0),
                "safety_settings": safety,
            },
            {
                "thinking_config": types.ThinkingConfig(thinking_budget=0),
                "safety_settings": safety,
            },
        ),
        "safety_mapping": (
            {"safety_settings": {harassment: only_high}},
            {"safety_settings": safety},
        ),
    }[name]


@pytest.mark.parametrize(
    "case",
    ["openai_options", "precedence", "thinking_and_safety_list", "safety_mapping"],
)
@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
def test_raw_request_keeps_generation_options(mode: Mode, case: str) -> None:
    types = pytest.importorskip("google.genai.types")
    handler_cls = (
        GenAIToolsHandler if mode == Mode.TOOLS else GenAIStructuredOutputsHandler
    )
    extra, expected = _raw_request_case(case, types)

    _, request = handler_cls(mode=mode).prepare_request(
        None, {"messages": [{"role": "user", "content": "Say hello"}], **extra}
    )

    config = request["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert {key: getattr(config, key) for key in expected} == expected
    assert not (set(extra) - {"config"}) & set(request)


class StructuredAnswer(BaseModel):
    answer: int


@pytest.mark.parametrize("config_kind", ["model", "dict"])
@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
def test_structured_request_keeps_config_options(mode: Mode, config_kind: str) -> None:
    types = pytest.importorskip("google.genai.types")
    handler_cls = (
        GenAIToolsHandler if mode == Mode.TOOLS else GenAIStructuredOutputsHandler
    )
    harassment = types.HarmCategory.HARM_CATEGORY_HARASSMENT
    only_high = types.HarmBlockThreshold.BLOCK_ONLY_HIGH
    safety = [types.SafetySetting(category=harassment, threshold=only_high)]
    thinking = types.ThinkingConfig(thinking_budget=512)

    config_fields: dict[str, Any] = {
        "temperature": 0.4,
        "max_output_tokens": 128,
        "top_p": 0.85,
        "top_k": 20,
        "seed": 7,
        "stop_sequences": ["END"],
        "safety_settings": safety,
        "system_instruction": "Be concise.",
        "thinking_config": thinking,
    }
    if mode == Mode.TOOLS:
        config_fields["tools"] = [
            types.Tool(
                function_declarations=[types.FunctionDeclaration(name="OtherTool")]
            )
        ]
        config_fields["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.NONE
            )
        )
    else:
        config_fields["response_mime_type"] = "text/plain"
        config_fields["response_schema"] = {"type": "string"}

    user_config = (
        types.GenerateContentConfig(**config_fields)
        if config_kind == "model"
        else config_fields
    )

    prepared_model, request = handler_cls(mode=mode).prepare_request(
        StructuredAnswer,
        {
            "messages": [{"role": "user", "content": "Say hello"}],
            "config": user_config,
        },
    )

    config = request["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.temperature == 0.4
    assert config.max_output_tokens == 128
    assert config.top_p == 0.85
    assert config.top_k == 20
    assert config.seed == 7
    assert config.stop_sequences == ["END"]
    assert config.safety_settings == safety
    assert config.system_instruction == "Be concise."
    assert config.thinking_config == thinking
    if mode == Mode.TOOLS:
        assert config.tools is not None
        assert config.tools[0].function_declarations is not None
        assert config.tools[0].function_declarations[0].name == "StructuredAnswer"
        assert config.tool_config is not None
        assert config.tool_config.function_calling_config is not None
        assert config.tool_config.function_calling_config.allowed_function_names == [
            "StructuredAnswer"
        ]
    else:
        assert config.response_mime_type == "application/json"
        assert config.response_schema is prepared_model


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
def test_structured_request_config_precedence_and_fallbacks(mode: Mode) -> None:
    types = pytest.importorskip("google.genai.types")
    handler_cls = (
        GenAIToolsHandler if mode == Mode.TOOLS else GenAIStructuredOutputsHandler
    )
    top_thinking = types.ThinkingConfig(thinking_budget=1024)
    config_thinking = types.ThinkingConfig(thinking_budget=64)

    _, request = handler_cls(mode=mode).prepare_request(
        StructuredAnswer,
        {
            "messages": [{"role": "user", "content": "Say hello"}],
            "system": "Explicit system prompt.",
            "generation_config": {
                "max_tokens": 8,
                "temperature": 0.3,
                "top_p": 0.6,
                "seed": 1,
            },
            "temperature": 0.9,
            "seed": 5,
            "thinking_config": top_thinking,
            "config": types.GenerateContentConfig(
                max_output_tokens=32,
                temperature=0.2,
                system_instruction="Ignored fallback.",
                thinking_config=config_thinking,
            ),
        },
    )

    config = request["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.max_output_tokens == 32
    assert config.temperature == 0.2
    assert config.top_p == 0.6
    assert config.seed == 5
    assert config.system_instruction == "Explicit system prompt."
    assert config.thinking_config == top_thinking


def test_update_genai_kwargs_preserves_caller_generation_config_and_handles_none() -> (
    None
):
    pytest.importorskip("google.genai.types")
    caller_generation_config = {"max_tokens": 100, "temperature": 0.7}
    kwargs = {"generation_config": caller_generation_config}

    result = update_genai_kwargs(kwargs, {})

    assert result["max_output_tokens"] == 100
    assert result["temperature"] == 0.7
    assert caller_generation_config == {"max_tokens": 100, "temperature": 0.7}

    none_result = update_genai_kwargs({"generation_config": None}, {})
    assert "safety_settings" in none_result

    ns_result = update_genai_kwargs(
        {
            "config": SimpleNamespace(
                temperature=0.0, labels={"env": "test"}, tools=None
            )
        },
        {},
    )
    assert ns_result["temperature"] == 0.0
    assert ns_result["labels"] == {"env": "test"}
    assert "tools" not in ns_result


def test_update_genai_kwargs_merges_config_fields_and_respects_owned_base_keys() -> (
    None
):
    types = pytest.importorskip("google.genai.types")

    harassment = types.HarmCategory.HARM_CATEGORY_HARASSMENT
    only_high = types.HarmBlockThreshold.BLOCK_ONLY_HIGH
    safety_list = [types.SafetySetting(category=harassment, threshold=only_high)]
    http_options = types.HttpOptions(timeout=5000)

    user_config = types.GenerateContentConfig(
        temperature=0.35,
        max_output_tokens=256,
        top_p=0.9,
        top_k=40,
        candidate_count=1,
        stop_sequences=["STOP"],
        seed=11,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        safety_settings=safety_list,
        system_instruction="Fallback system instruction",
        http_options=http_options,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
        tools=[
            types.Tool(function_declarations=[types.FunctionDeclaration(name="User")])
        ],
    )
    owned_tools = [
        types.Tool(function_declarations=[types.FunctionDeclaration(name="Owned")])
    ]
    base_config = {
        "tools": owned_tools,
        "system_instruction": None,
    }

    result = update_genai_kwargs({"config": user_config}, base_config)

    assert result["temperature"] == 0.35
    assert result["max_output_tokens"] == 256
    assert result["top_p"] == 0.9
    assert result["top_k"] == 40
    assert result["candidate_count"] == 1
    assert result["stop_sequences"] == ["STOP"]
    assert result["seed"] == 11
    assert result["presence_penalty"] == 0.1
    assert result["frequency_penalty"] == 0.2
    assert result["safety_settings"] == safety_list
    assert result["system_instruction"] == "Fallback system instruction"
    assert result["http_options"] is http_options
    assert result["media_resolution"] == types.MediaResolution.MEDIA_RESOLUTION_LOW
    assert result["tools"] is owned_tools


def test_update_genai_kwargs_safety_settings_priority_with_config() -> None:
    types = pytest.importorskip("google.genai.types")

    harassment = types.HarmCategory.HARM_CATEGORY_HARASSMENT
    hate = types.HarmCategory.HARM_CATEGORY_HATE_SPEECH
    only_high = types.HarmBlockThreshold.BLOCK_ONLY_HIGH
    block_low = types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE

    from_dict_config = update_genai_kwargs(
        {"config": {"safety_settings": {harassment: only_high}}},
        {},
    )
    by_cat = {
        entry["category"]: entry["threshold"]
        for entry in from_dict_config["safety_settings"]
    }
    assert by_cat[harassment] == only_high
    assert by_cat[hate] == types.HarmBlockThreshold.OFF

    top_list = [types.SafetySetting(category=hate, threshold=block_low)]
    top_wins = update_genai_kwargs(
        {
            "safety_settings": top_list,
            "config": {"safety_settings": {harassment: only_high}},
        },
        {},
    )
    assert top_wins["safety_settings"] == top_list
