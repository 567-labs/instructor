from __future__ import annotations

import builtins
import sys
from collections.abc import AsyncGenerator, Generator, Iterable
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from google.genai import types
from pydantic import BaseModel, ValidationError

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.errors import (
    ClientError,
    ConfigurationError,
    ModeError,
    ResponseParsingError,
)
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.dsl.iterable import IterableBase, IterableModel
from instructor.v2.dsl.partial import Partial, PartialBase
from instructor.v2.dsl.simple_type import ModelAdapter
from instructor.v2.providers import gemini
from instructor.v2.providers.gemini import client as gemini_client
from instructor.v2.providers.gemini import handlers, schema, templating, utils


class Answer(BaseModel):
    """A small structured answer returned by Gemini."""

    value: int


class LegacyModel:
    def __init__(self) -> None:
        self.sync_requests: list[dict[str, Any]] = []
        self.async_requests: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.sync_requests.append(kwargs)
        return SimpleNamespace(text='{"value": 7}')

    async def generate_content_async(self, **kwargs: Any) -> Any:
        self.async_requests.append(kwargs)
        return SimpleNamespace(text='{"value": 9}')


class FunctionCall:
    def __init__(self, *, name: str = "Answer", args: Any = None) -> None:
        self.name = name
        self.args = args

    @classmethod
    def to_dict(cls, call: FunctionCall) -> dict[str, Any]:
        return {"name": call.name, "args": call.args}


class DictOnlyFunctionCall:
    args = None

    @classmethod
    def to_dict(cls, _call: DictOnlyFunctionCall) -> dict[str, Any]:
        return {"name": "Answer", "args": {"value": "11"}}


class BrokenFunctionCall:
    args = None

    @classmethod
    def to_dict(cls, _call: BrokenFunctionCall) -> dict[str, Any]:
        raise ValueError("invalid function-call payload")


class NameOnlyFunctionCall:
    @classmethod
    def to_dict(cls, _call: NameOnlyFunctionCall) -> dict[str, Any]:
        return {"name": "Answer"}


class UnexpectedFunctionCallError:
    args = None

    @classmethod
    def to_dict(cls, _call: UnexpectedFunctionCallError) -> dict[str, Any]:
        raise RuntimeError("unexpected conversion failure")


class UnexpectedCompletionError:
    @property
    def candidates(self) -> list[Any]:
        raise RuntimeError("unexpected candidate failure")


class Part:
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


class Completion:
    def __init__(
        self,
        *,
        text: str | None = None,
        part_text: str | None = None,
        function_call: Any = None,
        text_error: type[Exception] | None = None,
    ) -> None:
        self._text = text
        self._text_error = text_error
        part = Part(text=part_text, function_call=function_call)
        self.parts = [part]
        self.candidates = [SimpleNamespace(content=SimpleNamespace(parts=[part]))]

    @property
    def text(self) -> str | None:
        if self._text_error is not None:
            raise self._text_error("response text is unavailable")
        return self._text


def _install_legacy_types(monkeypatch: pytest.MonkeyPatch) -> None:
    class FunctionResponse:
        def __init__(self, *, name: str, response: dict[str, Any]) -> None:
            self.name = name
            self.response = response

    class FunctionDeclaration:
        def __init__(
            self,
            *,
            name: str,
            description: str,
            parameters: dict[str, Any],
        ) -> None:
            self.name = name
            self.description = description
            self.parameters = parameters

    glm = ModuleType("google.ai.generativelanguage")
    vars(glm)["FunctionCall"] = FunctionCall
    vars(glm)["FunctionResponse"] = FunctionResponse
    vars(glm)["Part"] = Part

    google_ai = ModuleType("google.ai")
    vars(google_ai)["generativelanguage"] = glm

    legacy_types = ModuleType("google.generativeai.types")
    vars(legacy_types)["FunctionDeclaration"] = FunctionDeclaration
    legacy = ModuleType("google.generativeai")
    vars(legacy)["types"] = legacy_types

    monkeypatch.setitem(sys.modules, "google.ai", google_ai)
    monkeypatch.setitem(sys.modules, "google.ai.generativelanguage", glm)
    monkeypatch.setitem(sys.modules, "google.generativeai", legacy)
    monkeypatch.setitem(sys.modules, "google.generativeai.types", legacy_types)


def test_gemini_package_exports_factory_lazily_and_rejects_unknown_name() -> None:
    assert gemini.__getattr__("from_gemini") is gemini_client.from_gemini

    with pytest.raises(AttributeError, match="missing_factory"):
        gemini.__getattr__("missing_factory")


def test_gemini_factory_reports_unsupported_mode_and_missing_or_invalid_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ModeError) as mode_error:
        gemini_client.from_gemini(object(), mode=Mode.JSON_SCHEMA)
    assert mode_error.value.provider == Provider.GEMINI.value
    assert Mode.JSON_SCHEMA.value == mode_error.value.mode
    assert Mode.TOOLS.value in mode_error.value.valid_modes

    monkeypatch.setattr(gemini_client, "genai", None)
    with pytest.raises(ClientError, match="google-generativeai is not installed"):
        gemini_client.from_gemini(object())

    monkeypatch.setattr(
        gemini_client, "genai", SimpleNamespace(GenerativeModel=LegacyModel)
    )
    with pytest.raises(ClientError, match="Got: object"):
        gemini_client.from_gemini(object())

    monkeypatch.setattr(gemini_client, "genai", SimpleNamespace())
    with pytest.raises(ClientError, match="genai.GenerativeModel"):
        gemini_client.from_gemini(LegacyModel())


@pytest.mark.asyncio
async def test_gemini_factory_patches_sync_and_async_generation_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gemini_client, "genai", SimpleNamespace(GenerativeModel=LegacyModel)
    )

    sync_native = LegacyModel()
    sync_client = gemini_client.from_gemini(
        sync_native, mode=Mode.MD_JSON, trace_id="sync-request"
    )
    assert isinstance(sync_client, Instructor)
    assert sync_client.client is sync_native
    assert sync_client.provider is Provider.GEMINI
    assert sync_client.mode is Mode.MD_JSON

    sync_result = sync_client.create(
        response_model=Answer,
        messages=[{"role": "user", "content": "return seven"}],
        max_retries=1,
    )
    assert sync_result.value == 7
    assert sync_native.sync_requests[0]["trace_id"] == "sync-request"
    assert sync_native.sync_requests[0]["generation_config"]["response_mime_type"] == (
        "application/json"
    )
    assert sync_native.sync_requests[0]["contents"][0]["parts"][-1] == "return seven"

    async_native = LegacyModel()
    async_client = gemini_client.from_gemini(
        async_native, mode=Mode.MD_JSON, use_async=True, trace_id="async-request"
    )
    assert isinstance(async_client, AsyncInstructor)
    assert async_client.client is async_native
    assert async_client.provider is Provider.GEMINI
    assert async_client.mode is Mode.MD_JSON

    async_result = await async_client.create(
        response_model=Answer,
        messages=[{"role": "user", "content": "return nine"}],
        max_retries=1,
    )
    assert async_result.value == 9
    assert async_native.async_requests[0]["trace_id"] == "async-request"
    assert async_native.async_requests[0]["contents"][0]["parts"][-1] == "return nine"


def test_legacy_schema_generation_maps_optional_and_enum_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_legacy_types(monkeypatch)
    schema.generate_gemini_schema.cache_clear()

    class Choice(BaseModel):
        """A documented legacy function."""

        status: str | None
        category: str

    with pytest.warns(DeprecationWarning, match="generate_gemini_schema is deprecated"):
        declaration = schema.generate_gemini_schema(Choice)

    assert declaration.name == "Choice"
    assert declaration.description == "A documented legacy function."
    assert declaration.parameters["properties"]["status"]["nullable"] is True
    assert declaration.parameters["properties"]["status"]["type"] == "string"
    assert set(declaration.parameters["required"]) == {"status", "category"}
    schema.generate_gemini_schema.cache_clear()


def test_legacy_schema_generation_gives_an_actionable_missing_sdk_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema.generate_gemini_schema.cache_clear()

    def missing_legacy_types(_name: str) -> Any:
        raise ImportError("legacy Gemini SDK missing")

    monkeypatch.setattr(schema.importlib, "import_module", missing_legacy_types)
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ImportError, match="Please install google-genai instead"):
            schema.generate_gemini_schema(Answer)
    schema.generate_gemini_schema.cache_clear()


def test_gemini_reask_messages_preserve_bad_arguments_and_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_legacy_types(monkeypatch)
    original = [{"role": "user", "parts": ["extract a value"]}]
    response = Completion(
        function_call=FunctionCall(name="Answer", args={"value": "bad"})
    )
    error = ValueError("value must be an integer")

    result = handlers.GeminiToolsHandler().handle_reask(
        {"contents": original.copy()}, response, error
    )

    assert result["contents"][0] == original[0]
    model_call = result["contents"][1]
    assert model_call["role"] == "model"
    assert model_call["parts"][0].name == "Answer"
    assert model_call["parts"][0].args == {"value": "bad"}
    function_result = result["contents"][2]["parts"][0].function_response
    assert function_result.name == "Answer"
    assert function_result.response == {
        "error": "Validation Error(s) found:\nvalue must be an integer"
    }
    assert result["contents"][3]["role"] == "user"
    assert "fix the errors" in result["contents"][3]["parts"][0]

    json_response = Completion(text='{"value": "bad"}')
    json_result = handlers.GeminiJSONHandler().handle_reask(
        {"contents": original.copy()}, json_response, error
    )
    assert json_result["contents"][-1]["role"] == "user"
    assert '{"value": "bad"}' in json_result["contents"][-1]["parts"][0]
    assert "value must be an integer" in json_result["contents"][-1]["parts"][0]


def test_gemini_parsers_handle_strict_json_and_blocked_or_bad_tool_responses() -> None:
    strict_response = Completion(text='```json\n{"value": 3}\n```')
    parsed = handlers.parse_gemini_json(Answer, strict_response, strict=True)
    assert isinstance(parsed, Answer)
    assert parsed.value == 3

    with pytest.raises(ValidationError, match="valid integer"):
        handlers.parse_gemini_json(
            Answer, Completion(text='{"value": "3"}'), strict=True
        )

    blocked = Completion(text_error=ValueError)
    with pytest.raises(ResponseParsingError) as blocked_error:
        handlers.parse_gemini_json(Answer, blocked)
    assert blocked_error.value.mode == "GEMINI_JSON"
    assert blocked_error.value.raw_response is blocked

    fallback = Completion(function_call=DictOnlyFunctionCall())
    fallback_result = handlers.parse_gemini_tools(Answer, fallback, strict=False)
    assert isinstance(fallback_result, Answer)
    assert fallback_result.value == 11

    missing_call = SimpleNamespace(candidates=[])
    with pytest.raises(ResponseParsingError, match="No tool call found") as call_error:
        handlers.parse_gemini_tools(Answer, missing_call)
    assert call_error.value.mode == "GEMINI_TOOLS"
    assert call_error.value.raw_response is missing_call

    broken_args = Completion(function_call=BrokenFunctionCall())
    with pytest.raises(
        ResponseParsingError, match="No tool call args found"
    ) as args_error:
        handlers.parse_gemini_tools(Answer, broken_args)
    assert args_error.value.mode == "GEMINI_TOOLS"
    assert args_error.value.raw_response is broken_args

    with pytest.raises(RuntimeError, match="unexpected candidate failure"):
        handlers.parse_gemini_tools(Answer, UnexpectedCompletionError())

    with pytest.raises(RuntimeError, match="unexpected conversion failure"):
        handlers.parse_gemini_tools(
            Answer,
            Completion(function_call=UnexpectedFunctionCallError()),
        )


def test_gemini_handlers_prepare_expected_tool_and_json_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Answer, "gemini_schema", {"name": "Answer"}, raising=False)
    original = {"messages": [{"role": "user", "content": "extract 4"}]}

    tools_model, tools_request = handlers.GeminiToolsHandler().prepare_request(
        Answer, original
    )
    json_model, json_request = handlers.GeminiJSONHandler().prepare_request(
        Answer, original
    )

    assert tools_model is Answer
    assert tools_request["tools"] == [{"name": "Answer"}]
    assert tools_request["tool_config"]["function_calling_config"] == {
        "mode": "ANY",
        "allowed_function_names": ["Answer"],
    }
    assert json_model is Answer
    assert json_request["generation_config"]["response_mime_type"] == "application/json"
    # The schema instruction is hoisted into the outgoing Gemini contents, and
    # the caller's own `messages` stays exactly as it was passed in.
    assert "json_schema" in json_request["contents"][0]["parts"][0]
    assert json_request["contents"][0]["parts"][1] == "extract 4"
    assert original == {"messages": [{"role": "user", "content": "extract 4"}]}


def test_gemini_tools_handler_parses_and_finalizes_a_valid_tool_response() -> None:
    response = Completion(function_call=FunctionCall(args={"value": 12}))

    parsed = handlers.GeminiToolsHandler().parse_response(response, Answer, strict=True)

    assert parsed.value == 12
    assert parsed._raw_response is response


def test_gemini_stream_extractors_keep_valid_chunks_and_skip_incomplete_chunks() -> (
    None
):
    tool_handler = handlers.GeminiToolsHandler()
    tool_chunks = [
        Completion(function_call=FunctionCall(args={"value": 1})),
        Completion(function_call=NameOnlyFunctionCall()),
        SimpleNamespace(),
    ]
    assert list(tool_handler.extract_streaming_json(tool_chunks)) == ['{"value": 1}']

    json_handler = handlers.GeminiJSONHandler()
    json_chunks = [
        Completion(text='{"value":'),
        Completion(part_text=" 2}", text_error=AttributeError),
        Completion(part_text="", text_error=AttributeError),
        SimpleNamespace(),
    ]
    assert list(json_handler.extract_streaming_json(json_chunks)) == [
        '{"value":',
        " 2}",
    ]


@pytest.mark.asyncio
async def test_gemini_async_stream_extractors_keep_valid_chunks_and_skip_incomplete_chunks() -> (
    None
):
    async def stream(chunks: Iterable[Any]) -> AsyncGenerator[Any, None]:
        for chunk in chunks:
            yield chunk

    tool_chunks = [
        Completion(function_call=FunctionCall(args={"value": 1})),
        Completion(function_call=NameOnlyFunctionCall()),
        SimpleNamespace(),
    ]
    assert [
        chunk
        async for chunk in handlers.GeminiToolsHandler().extract_streaming_json_async(
            stream(tool_chunks)
        )
    ] == ['{"value": 1}']

    json_chunks = [
        Completion(text='{"value":'),
        Completion(part_text=" 2}", text_error=AttributeError),
        Completion(part_text="", text_error=AttributeError),
        SimpleNamespace(),
    ]
    assert [
        chunk
        async for chunk in handlers.GeminiJSONHandler().extract_streaming_json_async(
            stream(json_chunks)
        )
    ] == ['{"value":', " 2}"]


@pytest.mark.asyncio
async def test_gemini_handlers_parse_iterable_partial_and_async_streams() -> None:
    iterable_model = IterableModel(Answer)
    assert issubclass(iterable_model, IterableBase)
    tools_handler = handlers.GeminiToolsHandler()
    tools_stream = [
        Completion(function_call=FunctionCall(args={"tasks": [{"value": 1}]}))
    ]
    tools_result = tools_handler.parse_response(
        tools_stream,
        iterable_model,
        validation_context={"source": "tool-stream"},
        strict=True,
        stream=True,
    )
    assert isinstance(tools_result, Generator)
    assert [item.value for item in tools_result] == [1]

    partial_model = Partial[Answer]
    assert issubclass(partial_model, PartialBase)
    partial_result = handlers.GeminiJSONHandler().parse_response(
        [Completion(text='{"value":'), Completion(text=" 2}")],
        partial_model,
        stream=True,
    )
    assert isinstance(partial_result, list)
    assert partial_result[-1].value == 2

    async def async_stream() -> AsyncGenerator[Any, None]:
        yield Completion(text='{"tasks": [{"value": 4}]}')

    async_result = handlers.GeminiJSONHandler().parse_response(
        async_stream(),
        iterable_model,
        validation_context={"source": "async-stream"},
        strict=True,
        stream=True,
    )
    assert [item.value async for item in async_result] == [4]


def test_gemini_stream_parser_supports_custom_models_and_unwraps_simple_types() -> None:
    class CustomStreamModel(BaseModel):
        value: int

        @classmethod
        def from_streaming_response(
            cls,
            response: Iterable[Any],
            stream_extractor: Any,
            **kwargs: Any,
        ) -> Generator[CustomStreamModel, None, None]:
            assert kwargs == {"context": {"source": "custom"}, "strict": False}
            text = "".join(stream_extractor(response))
            yield cls.model_validate_json(text)

    handler = handlers.GeminiJSONHandler()
    streamed = handler._parse_streaming(
        CustomStreamModel,
        [Completion(text='{"value": 5}')],
        validation_context={"source": "custom"},
        strict=False,
    )
    assert [item.value for item in streamed] == [5]

    adapted = cast(type[BaseModel], ModelAdapter[int])
    assert handler.parse_response(Completion(text='{"content": 6}'), adapted) == 6
    parsed = handler.parse_response(Completion(text='{"value": 8}'), Answer)
    assert parsed.value == 8
    assert parsed._raw_response.text == '{"value": 8}'
    raw_result = {"value": 9}
    assert handler._finalize(Answer, Completion(), raw_result) is raw_result


def test_gemini_schema_mapping_handles_enums_optional_and_supported_unions() -> None:
    mapped = utils.map_to_gemini_function_schema(
        {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ready", "done"]},
                "optional": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                },
                "score": {
                    "anyOf": [{"type": "string"}, {"type": "number"}],
                },
                "count": {
                    "anyOf": [{"type": "integer"}, {"type": "boolean"}],
                },
            },
            "required": ["status", "optional", "score"],
        }
    )

    status = mapped["properties"]["status"]
    assert status == {"enum": ["ready", "done"], "format": "enum", "type": "string"}
    assert mapped["properties"]["optional"] == {
        "nullable": True,
        "type": "string",
    }
    assert mapped["properties"]["score"]["anyOf"] == [
        {"type": "string"},
        {"type": "number"},
    ]
    assert mapped["properties"]["count"]["anyOf"] == [
        {"type": "integer"},
        {"type": "boolean"},
    ]


def test_gemini_schema_mapping_surfaces_unsupported_union_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils, "verify_no_unions", lambda _schema: False)
    with pytest.raises(ValueError, match="Gemini does not support Union types"):
        utils.map_to_gemini_function_schema(Answer.model_json_schema())


def test_gemini_safety_defaults_fall_back_to_legacy_sdk_or_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    class LegacyCategory:
        HARM_CATEGORY_HATE_SPEECH = "hate"
        HARM_CATEGORY_HARASSMENT = "harassment"
        HARM_CATEGORY_DANGEROUS_CONTENT = "dangerous"

    class LegacyThreshold:
        BLOCK_ONLY_HIGH = "high"

    legacy_module = ModuleType("google.generativeai.types")
    vars(legacy_module)["HarmCategory"] = LegacyCategory
    vars(legacy_module)["HarmBlockThreshold"] = LegacyThreshold

    def import_with_legacy(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name == "google.genai.types":
            raise ImportError("new GenAI SDK missing")
        if name == "google.generativeai.types":
            return legacy_module
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_legacy)
    utils._default_safety_thresholds.cache_clear()
    assert utils._default_safety_thresholds() == {
        "hate": "high",
        "harassment": "high",
        "dangerous": "high",
    }

    def import_without_google(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name in {"google.genai.types", "google.generativeai.types"}:
            raise ImportError("Google SDK missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_google)
    utils._default_safety_thresholds.cache_clear()
    assert utils._default_safety_thresholds() is None
    utils._default_safety_thresholds.cache_clear()


def test_gemini_model_schema_and_generation_config_accept_compatibility_shapes() -> (
    None
):
    class SchemaHolder:
        model_json_schema = {"type": "object", "properties": {}}

    assert utils._get_model_schema(SchemaHolder) == {
        "type": "object",
        "properties": {},
    }

    safety_setting = types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    )
    updated = utils.update_genai_kwargs(
        {"safety_settings": [safety_setting]},
        {"response_mime_type": "application/json"},
    )
    assert updated["response_mime_type"] == "application/json"
    assert updated["safety_settings"] == [safety_setting]

    config_object = SimpleNamespace(
        thinking_config={"thinking_budget": 10}, labels={"suite": "coverage"}
    )
    inherited = utils.update_genai_kwargs({"config": config_object}, {})
    assert inherited["thinking_config"] == {"thinking_budget": 10}
    assert inherited["labels"] == {"suite": "coverage"}
    assert "cached_content" not in inherited

    with_null_token_limit = utils.update_genai_kwargs(
        {"generation_config": {"max_tokens": None, "temperature": 0.25}}, {}
    )
    assert "max_output_tokens" not in with_null_token_limit
    assert with_null_token_limit["temperature"] == 0.25

    default_thresholds = utils.update_genai_kwargs({"safety_settings": ()}, {})
    assert default_thresholds["safety_settings"]
    assert all(
        setting["threshold"] is types.HarmBlockThreshold.OFF
        for setting in default_thresholds["safety_settings"]
    )

    legacy_kwargs = utils.update_gemini_kwargs(
        {"generation_config": {"max_tokens": None, "temperature": 0.5}}
    )
    assert "contents" not in legacy_kwargs
    assert "max_output_tokens" not in legacy_kwargs["generation_config"]
    assert legacy_kwargs["generation_config"]["temperature"] == 0.5


def test_genai_message_conversion_rejects_invalid_roles_parts_and_message_types() -> (
    None
):
    assert (
        utils.extract_genai_system_message(
            cast(
                list[dict[str, Any]],
                [
                    "raw prompt",
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "hi"},
                ],
            )
        )
        == "rules\n\n"
    )
    assert utils.transform_to_gemini_prompt([]) == []
    assert utils.transform_to_gemini_prompt(
        cast(
            Any,
            [
                {"role": "system", "content": None},
                {
                    "role": "system",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "image.png"}},
                        {"type": "text", "text": 42},
                        {"type": "text", "text": "use integers"},
                    ],
                },
                {"role": "user", "content": "extract 4"},
            ],
        )
    ) == [{"role": "user", "parts": ["*use integers*", "extract 4"]}]
    assert (
        utils.extract_genai_system_message(
            cast(
                list[dict[str, Any]],
                [
                    42,
                    {"role": "system", "content": None},
                    {"role": "system", "content": ["use integers", object()]},
                    {"role": "user", "content": "extract 4"},
                ],
            )
        )
        == "use integers\n\n"
    )
    assert (
        utils.convert_to_genai_messages(
            [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "hi"},
            ]
        )[0]
        .parts[0]
        .text
        == "hi"
    )

    with pytest.raises(ValueError, match="Unsupported role: assistant"):
        utils.convert_to_genai_messages(
            [{"role": "assistant", "content": "not a GenAI role"}]
        )

    with pytest.raises(ValueError, match="Unsupported content item type"):
        utils.convert_to_genai_messages([{"role": "user", "content": [object()]}])

    with pytest.raises(ValueError, match="Unsupported content type"):
        utils.convert_to_genai_messages([{"role": "user", "content": 42}])

    with pytest.raises(ValueError, match="Unsupported message type"):
        utils.convert_to_genai_messages([cast(Any, 42)])


def test_gemini_request_helpers_support_unstructured_requests_and_reject_model_override() -> (
    None
):
    json_model, json_kwargs = utils.handle_gemini_json(
        None,
        {
            "messages": [{"role": "user", "content": "hello"}],
            "generation_config": {"max_tokens": 8},
        },
    )
    assert json_model is None
    assert json_kwargs["contents"] == [{"role": "user", "parts": ["hello"]}]
    assert json_kwargs["generation_config"]["max_output_tokens"] == 8

    with pytest.raises(ConfigurationError, match="must be set while patching"):
        utils.handle_gemini_tools(Answer, {"model": "gemini-pro", "messages": []})

    tools_model, tools_kwargs = utils.handle_gemini_tools(
        None, {"messages": [{"role": "user", "content": "hello"}]}
    )
    assert tools_model is None
    assert tools_kwargs["contents"] == [{"role": "user", "parts": ["hello"]}]

    _, existing_system = utils.handle_gemini_json(
        Answer,
        {
            "messages": [
                {"role": "system", "content": "keep it brief"},
                {"role": "user", "content": "hello"},
            ]
        },
    )
    assert "keep it brief" in existing_system["contents"][0]["parts"][0]
    assert "json_schema" in existing_system["contents"][0]["parts"][0]


def test_genai_request_helpers_cover_unstructured_streaming_and_config_inheritance() -> (
    None
):
    no_model, unstructured = utils.handle_genai_structured_outputs(
        None,
        {
            "messages": [{"role": "user", "content": "hello"}],
            "generation_config": {"max_tokens": 16},
        },
    )
    assert no_model is None
    assert unstructured["contents"][0].parts[0].text == "hello"
    assert unstructured["config"].max_output_tokens == 16

    _, explicit_system = utils.handle_genai_structured_outputs(
        None,
        {
            "system": "use the explicit instructions",
            "messages": [
                {"role": "system", "content": "ignore these instructions"},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert explicit_system["config"].system_instruction == (
        "use the explicit instructions"
    )
    assert "system" not in explicit_system

    structured_model, structured = utils.handle_genai_structured_outputs(
        Answer,
        {
            "messages": [],
            "stream": True,
            "config": {"thinking_config": {"thinking_budget": 32}},
        },
    )
    assert structured_model is not None
    assert issubclass(structured_model, PartialBase)
    assert structured["config"].response_mime_type == "application/json"
    assert structured["config"].response_schema is structured_model
    assert structured["config"].thinking_config.thinking_budget == 32
    assert structured["config"].system_instruction is None

    tool_model, tool_request = utils.handle_genai_tools(
        Answer,
        {
            "messages": [],
            "stream": True,
            "config": {"thinking_config": {"thinking_budget": 64}},
        },
    )
    assert tool_model is not None
    assert issubclass(tool_model, PartialBase)
    declaration = tool_request["config"].tools[0].function_declarations[0]
    assert declaration.name == tool_model.__name__
    assert declaration.parameters.properties["value"].type is types.Type.INTEGER
    assert tool_request["config"].tool_config.function_calling_config.mode is (
        types.FunctionCallingConfigMode.ANY
    )
    assert tool_request["config"].thinking_config.thinking_budget == 64
    assert tool_request["config"].system_instruction is None


def test_genai_request_helpers_preserve_cached_content_and_system_messages_from_config() -> (
    None
):
    cached = types.GenerateContentConfig(
        cached_content="cachedContents/session-1",
        thinking_config=types.ThinkingConfig(thinking_budget=24),
    )
    messages = [
        {"role": "system", "content": "use cached rules"},
        {"role": "user", "content": "extract a value"},
    ]

    structured_model, structured = utils.handle_genai_structured_outputs(
        Answer, {"messages": messages.copy(), "config": cached}
    )
    assert structured_model is Answer
    assert structured["config"].cached_content == "cachedContents/session-1"
    assert structured["config"].thinking_config.thinking_budget == 24
    assert structured["config"].system_instruction is None
    assert structured["contents"][0].parts[0].text == "extract a value"

    tool_model, tool_request = utils.handle_genai_tools(
        Answer, {"messages": messages.copy(), "config": cached}
    )
    assert tool_model is Answer
    assert tool_request["config"].cached_content == "cachedContents/session-1"
    assert tool_request["config"].thinking_config.thinking_budget == 24
    assert tool_request["config"].system_instruction is None
    assert tool_request["config"].tools is None
    assert tool_request["contents"][0].parts[0].text == "extract a value"

    cached_only = SimpleNamespace(cached_content="cachedContents/session-2")
    _, structured_cached_only = utils.handle_genai_structured_outputs(
        Answer, {"messages": messages.copy(), "config": cached_only}
    )
    _, tools_cached_only = utils.handle_genai_tools(
        Answer, {"messages": messages.copy(), "config": cached_only}
    )
    assert structured_cached_only["config"].cached_content == "cachedContents/session-2"
    assert structured_cached_only["config"].thinking_config is None
    assert tools_cached_only["config"].cached_content == "cachedContents/session-2"
    assert tools_cached_only["config"].thinking_config is None

    thinking_only = SimpleNamespace(
        thinking_config=types.ThinkingConfig(thinking_budget=12)
    )
    _, structured_thinking_only = utils.handle_genai_structured_outputs(
        Answer, {"messages": messages.copy(), "config": thinking_only}
    )
    _, tools_thinking_only = utils.handle_genai_tools(
        Answer, {"messages": messages.copy(), "config": thinking_only}
    )
    assert structured_thinking_only["config"].cached_content is None
    assert structured_thinking_only["config"].thinking_config.thinking_budget == 12
    assert tools_thinking_only["config"].cached_content is None
    assert tools_thinking_only["config"].thinking_config.thinking_budget == 12


def test_gemini_compatibility_helpers_keep_vertex_passthrough_and_genai_reask() -> None:
    kwargs = {"messages": [{"role": "user", "content": "hello"}]}
    vertex_model, vertex_tools = utils.handle_vertexai_tools(None, kwargs)
    json_model, vertex_json = utils.handle_vertexai_json(None, kwargs)
    assert vertex_model is None
    assert vertex_tools is kwargs
    assert json_model is None
    assert vertex_json is kwargs

    call = types.FunctionCall(name="Answer", args={"value": "wrong"})
    model_content = types.Content(role="model", parts=[types.Part(function_call=call)])
    response = SimpleNamespace(candidates=[SimpleNamespace(content=model_content)])
    reasked = utils.reask_genai_tools(
        {
            "contents": [
                types.Content(role="user", parts=[types.Part.from_text(text="hi")])
            ]
        },
        response,
        ValueError("value must be an integer"),
    )
    assert reasked["contents"][1] is model_content
    function_response = reasked["contents"][2].parts[0].function_response
    assert function_response.name == "Answer"
    assert "value must be an integer" in function_response.response["error"]


def test_gemini_templating_updates_text_parts_and_preserves_media_parts() -> None:
    media = object()
    message = {"role": "user", "parts": ["hello {{ name }}", media]}

    result = templating.process_message(
        message,
        {"name": "Gemini"},
        lambda text, context: text.replace("{{ name }}", context["name"]),
    )

    assert result is message
    assert result["parts"] == ["hello Gemini", media]
    assert templating.process_message({"role": "user"}, {}, lambda text, _: text) == {
        "role": "user"
    }
