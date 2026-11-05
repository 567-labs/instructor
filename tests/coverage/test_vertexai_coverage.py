"""Offline coverage for the VertexAI v2 client, handlers, and helpers."""

from __future__ import annotations

import builtins
import importlib
import runpy
import sys
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any, Union, cast

import pytest
import vertexai.generative_models as gm
from pydantic import BaseModel

from instructor import Mode
from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.errors import ClientError, ModeError
from instructor.v2.core.providers import Provider
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.partial import Partial
from instructor.v2.dsl.simple_type import ModelAdapter
from instructor.v2.providers.vertexai import handlers, templating
from instructor.v2.providers.vertexai.parallel import (
    VertexAIParallelBase,
    VertexAIParallelModel,
)
from tests.coverage._streams import async_items

vertex_client = importlib.import_module("instructor.v2.providers.vertexai.client")


class Weather(BaseModel):
    """Weather requested for a city."""

    city: str


class Score(BaseModel):
    """A numeric score."""

    value: int


def _part(
    *,
    name: str = "Weather",
    args: dict[str, Any] | None = None,
    text: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        function_call=SimpleNamespace(name=name, args=args or {}),
        text=text,
    )


def _response(*parts: Any, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))],
    )


def test_vertexai_lazy_exports_resolve_and_unknown_exports_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("instructor.v2.providers.vertexai")

    for name in package.__all__:
        monkeypatch.delitem(package.__dict__, name, raising=False)
        resolved = getattr(package, name)
        module_path, attr_name = package._LAZY_ATTRS[name]
        expected = getattr(
            importlib.import_module(module_path, package.__name__), attr_name
        )
        assert resolved is expected
        assert package.__dict__[name] is expected

    with pytest.raises(AttributeError, match="not_exported"):
        package.__getattr__("not_exported")


def test_vertexai_client_without_optional_sdk_has_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "vertexai.generative_models", None)

    isolated = runpy.run_path(vertex_client.__file__, run_name="vertex_client_no_sdk")

    assert isolated["gm"] is None
    with pytest.raises(ClientError, match="pip install google-cloud-aiplatform"):
        isolated["from_vertexai"](object(), mode=Mode.TOOLS)


def test_vertexai_client_validates_mode_and_client_type() -> None:
    model = object.__new__(gm.GenerativeModel)

    with pytest.raises(ModeError) as unsupported:
        vertex_client.from_vertexai(model, mode=Mode.JSON)
    assert unsupported.value.mode == Mode.JSON.value
    assert unsupported.value.provider == Provider.VERTEXAI.value
    assert Mode.TOOLS.value in unsupported.value.valid_modes

    with pytest.raises(ClientError, match="Got: object"):
        vertex_client.from_vertexai(cast(gm.GenerativeModel, object()), mode=Mode.TOOLS)


def test_vertexai_client_selects_and_patches_sync_and_async_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object.__new__(gm.GenerativeModel)
    calls: list[tuple[str, Provider, Mode]] = []

    def patch(*, func: Any, provider: Provider, mode: Mode) -> Any:
        calls.append((func.__name__, provider, mode))
        return func

    monkeypatch.setattr(vertex_client, "patch_v2", patch)

    sync_client = vertex_client.from_vertexai(model, mode=Mode.TOOLS)
    async_client = vertex_client.from_vertexai(model, mode=Mode.MD_JSON, use_async=True)

    assert isinstance(sync_client, Instructor)
    assert isinstance(async_client, AsyncInstructor)
    assert sync_client.client is model
    assert async_client.client is model
    assert (sync_client.provider, sync_client.mode) == (Provider.VERTEXAI, Mode.TOOLS)
    assert (async_client.provider, async_client.mode) == (
        Provider.VERTEXAI,
        Mode.MD_JSON,
    )
    assert calls == [
        ("generate_content", Provider.VERTEXAI, Mode.TOOLS),
        ("generate_content_async", Provider.VERTEXAI, Mode.MD_JSON),
    ]


@pytest.mark.parametrize(
    ("blocked", "helper"),
    [
        ("vertexai.generative_models", "_gm"),
        ("vertexai.preview.generative_models", "_tool_config_cls"),
    ],
)
def test_vertexai_handler_import_errors_explain_optional_dependency(
    monkeypatch: pytest.MonkeyPatch, blocked: str, helper: str
) -> None:
    original_import = builtins.__import__

    def missing_sdk(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == blocked:
            raise ImportError(f"No module named {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_sdk)

    with pytest.raises(ImportError, match="pip install google-cloud-aiplatform"):
        getattr(handlers, helper)()


def test_vertexai_tools_build_one_declaration_per_parallel_model() -> None:
    tool = handlers._create_vertexai_tool(Iterable[Union[Weather, Score]])
    declarations = tool._raw_tool.function_declarations

    assert [declaration.name for declaration in declarations] == ["Weather", "Score"]
    assert declarations[0].description == "Weather requested for a city."
    assert declarations[1].description == "A numeric score."


def test_vertexai_handlers_prepare_tool_json_and_parallel_requests() -> None:
    request = {"messages": [{"role": "user", "content": "Weather in Paris?"}]}
    tools_handler = handlers.VertexAIToolsHandler()
    json_handler = handlers.VertexAIJSONHandler()
    parallel_handler = handlers.VertexAIParallelToolsHandler()

    tool_model, tool_request = tools_handler.prepare_request(Weather, request)
    json_model, json_request = json_handler.prepare_request(Weather, request)
    no_model, untouched = parallel_handler.prepare_request(None, request)
    parallel_model, parallel_request = parallel_handler.prepare_request(
        cast(type[BaseModel], Iterable[Weather]), request
    )

    assert tool_model is Weather
    assert tool_request["contents"][0].parts[0].text == "Weather in Paris?"
    assert tool_request["tools"][0]._raw_tool.function_declarations[0].name == "Weather"
    assert (
        tool_request["tool_config"]._gapic_tool_config.function_calling_config.mode.name
        == "ANY"
    )
    assert json_model is Weather
    assert json_request["contents"][0].parts[0].text == "Weather in Paris?"
    assert (
        json_request["generation_config"]._raw_generation_config.response_mime_type
        == "application/json"
    )
    assert no_model is None
    assert untouched == request
    assert isinstance(parallel_model, VertexAIParallelBase)
    assert list(parallel_model.registry) == ["Weather"]
    assert (
        parallel_request["tools"][0]._raw_tool.function_declarations[0].name
        == "Weather"
    )


def test_vertexai_handlers_parse_reask_and_finalize_responses() -> None:
    tools_handler = handlers.VertexAIToolsHandler()
    json_handler = handlers.VertexAIJSONHandler()
    parallel_handler = handlers.VertexAIParallelToolsHandler()
    tool_response = _response(_part(args={"city": "Paris"}))
    json_response = _response(text='{"city": "Paris"}')
    error = ValueError("city is required")

    parsed_tool = tools_handler.parse_response(
        tool_response, Weather, validation_context={"request_id": "tool"}
    )
    parsed_json = json_handler.parse_response(
        json_response,
        Weather,
        validation_context={"request_id": "json"},
        strict=True,
    )
    adapted = tools_handler.parse_response(
        _response(_part(name="Response", args={"content": 7})),
        cast(type[BaseModel], ModelAdapter[int]),
    )
    tool_retry = tools_handler.handle_reask({"contents": []}, tool_response, error)
    json_retry = json_handler.handle_reask({"contents": []}, json_response, error)
    parallel_retry = parallel_handler.handle_reask(
        {"contents": []}, tool_response, error
    )

    assert parsed_tool == Weather(city="Paris")
    assert parsed_tool._raw_response is tool_response
    assert parsed_json == Weather(city="Paris")
    assert parsed_json._raw_response is json_response
    assert adapted == 7
    assert len(tool_retry["contents"]) == 2
    assert tool_retry["contents"][1].parts[0].function_response.name == "Weather"
    assert "city is required" in json_retry["contents"][1].parts[0].text
    assert len(parallel_retry["contents"]) == 2


def test_vertexai_sync_stream_extractors_skip_empty_chunks() -> None:
    tools_handler = handlers.VertexAIToolsHandler()
    json_handler = handlers.VertexAIJSONHandler()
    tool_stream = [object(), _response(_part(args={"city": "Paris"}))]
    json_stream = [object(), _response(_part(text='{"city":"Paris"}'))]

    assert list(tools_handler.extract_streaming_json(tool_stream)) == [
        '{"city": "Paris"}'
    ]
    assert list(json_handler.extract_streaming_json(json_stream)) == [
        '{"city":"Paris"}'
    ]


@pytest.mark.asyncio
async def test_vertexai_async_stream_extractors_skip_empty_chunks() -> None:
    tools_handler = handlers.VertexAIToolsHandler()
    json_handler = handlers.VertexAIJSONHandler()
    tool_stream = [object(), _response(_part(args={"city": "Paris"}))]
    json_stream = [object(), _response(_part(text='{"city":"Paris"}'))]

    tool_chunks = [
        chunk
        async for chunk in tools_handler.extract_streaming_json_async(
            async_items(tool_stream)
        )
    ]
    json_chunks = [
        chunk
        async for chunk in json_handler.extract_streaming_json_async(
            async_items(json_stream)
        )
    ]

    assert tool_chunks == ['{"city": "Paris"}']
    assert json_chunks == ['{"city":"Paris"}']


@pytest.mark.asyncio
async def test_vertexai_streaming_parsers_return_sync_and_async_iterable_items() -> (
    None
):
    tools_handler = handlers.VertexAIToolsHandler()
    json_handler = handlers.VertexAIJSONHandler()
    weather_list = IterableModel(Weather)
    tool_stream = [
        _response(_part(args={"tasks": [{"city": "Paris"}, {"city": "London"}]}))
    ]
    json_stream = [_response(_part(text='{"tasks":[{"city":"Tokyo"}]}'))]

    sync_tools = tools_handler.parse_response(
        tool_stream,
        weather_list,
        stream=True,
        validation_context={"request_id": "sync"},
        strict=True,
    )
    async_tools = tools_handler.parse_response(
        async_items(tool_stream),
        weather_list,
        stream=True,
        validation_context={"request_id": "async-tool"},
        strict=True,
    )
    async_json = json_handler.parse_response(
        async_items(json_stream),
        weather_list,
        stream=True,
        validation_context={"request_id": "async-json"},
        strict=True,
    )
    default_json = json_handler.parse_response(json_stream, weather_list, stream=True)

    assert [item.city for item in sync_tools] == ["Paris", "London"]
    assert [item.city async for item in async_tools] == ["Paris", "London"]
    assert [item.city async for item in async_json] == ["Tokyo"]
    assert [item.city for item in default_json] == ["Tokyo"]


@pytest.mark.asyncio
async def test_vertexai_streaming_parser_materializes_partial_and_custom_models() -> (
    None
):
    handler = handlers.VertexAIJSONHandler()

    class StreamingWeather(BaseModel):
        city: str

        @classmethod
        def from_streaming_response(
            cls,
            completion: Iterable[Any],
            stream_extractor: Any,
            **kwargs: Any,
        ) -> Iterable[StreamingWeather]:
            assert kwargs == {"context": {"request_id": "custom"}, "strict": False}
            payload = "".join(stream_extractor(completion))
            yield cls.model_validate_json(payload, **kwargs)

    partial = handler.parse_response(
        [_response(_part(text='{"city":"Par')), _response(_part(text='is"}'))],
        Partial[Weather],
        stream=True,
        validation_context={"request_id": "partial"},
        strict=True,
    )
    async_partial = [
        item
        async for item in handler.parse_response(
            async_items(
                [_response(_part(text='{"city":"Lon')), _response(_part(text='don"}'))]
            ),
            Partial[Weather],
            stream=True,
            validation_context={"request_id": "async-partial"},
            strict=True,
        )
    ]
    custom = handler._parse_streaming(
        StreamingWeather,
        [_response(_part(text='{"city":"Oslo"}'))],
        validation_context={"request_id": "custom"},
        strict=False,
    )

    assert isinstance(partial, list)
    assert [item.city for item in partial] == ["Par", "Paris"]
    assert async_partial[-1].model_dump() == {"city": "London"}
    assert isinstance(custom, list)
    assert [item.city for item in custom] == ["Oslo"]


def test_vertexai_parallel_parsers_validate_known_calls_and_skip_empty_candidates() -> (
    None
):
    model = VertexAIParallelModel(Iterable[Union[Weather, Score]])
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(content=None),
            SimpleNamespace(content=SimpleNamespace(parts=[])),
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="No function call was requested."),
                        SimpleNamespace(function_call=None),
                        _part(name="Weather", args={"city": "Paris"}),
                        _part(name="Score", args={"value": 9}),
                    ]
                )
            ),
        ]
    )
    tools_handler = handlers.VertexAIToolsHandler()
    parallel_handler = handlers.VertexAIParallelToolsHandler()

    parsed_by_tools = list(
        tools_handler.parse_response(
            response,
            model,
            validation_context={"request_id": "tools"},
            strict=True,
        )
    )
    parsed_parallel = list(
        parallel_handler.parse_response(
            response,
            model,
            validation_context={"request_id": "parallel"},
            strict=True,
        )
    )
    single_response = _response(_part(name="Weather", args={"city": "Paris"}))
    parsed_single = parallel_handler.parse_response(single_response, Weather)

    assert parsed_by_tools == [Weather(city="Paris"), Score(value=9)]
    assert parsed_parallel == [Weather(city="Paris"), Score(value=9)]
    assert parsed_single == Weather(city="Paris")
    assert parsed_single._raw_response is single_response


def test_vertexai_message_templating_preserves_role_and_renders_each_part() -> None:
    message = gm.Content(
        role="user",
        parts=[
            gm.Part.from_text("Weather in {{ city }}?"),
            gm.Part.from_text("Use {{ unit }}."),
        ],
    )
    calls: list[tuple[str, dict[str, Any]]] = []

    def apply_template(value: str, context: dict[str, Any]) -> str:
        calls.append((value, context))
        return value.replace("{{ city }}", context["city"]).replace(
            "{{ unit }}", context["unit"]
        )

    rendered = templating.process_message(
        message, {"city": "Paris", "unit": "Celsius"}, apply_template
    )

    assert rendered.role == "user"
    assert [part.text for part in rendered.parts] == [
        "Weather in Paris?",
        "Use Celsius.",
    ]
    assert calls == [
        ("Weather in {{ city }}?", {"city": "Paris", "unit": "Celsius"}),
        ("Use {{ unit }}.", {"city": "Paris", "unit": "Celsius"}),
    ]
