"""Offline behavior coverage for the Google GenAI v2 provider."""

from __future__ import annotations

import builtins
import importlib
import json
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
from google.genai import Client, types
from pydantic import BaseModel

from instructor.v2.core.errors import ClientError, ModeError
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.partial import Partial, PartialBase
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.providers.genai import handlers, multimodal
from instructor.v2.providers.genai import client as genai_client
from tests.coverage._streams import async_items


class Answer(BaseModel):
    answer: int


class RecordingModels:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate_content(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("content", kwargs))
        return {"kind": "content", **kwargs}

    def generate_content_stream(self, **kwargs: Any) -> Iterator[dict[str, Any]]:
        self.calls.append(("stream", kwargs))
        yield {"kind": "stream", **kwargs}


class RecordingAsyncModels:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def generate_content(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("content", kwargs))
        return {"kind": "content", **kwargs}

    async def generate_content_stream(
        self, **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        self.calls.append(("stream", kwargs))

        async def stream() -> AsyncIterator[dict[str, Any]]:
            yield {"kind": "stream", **kwargs}

        return stream()


class RecordingClient:
    def __init__(self) -> None:
        self.models = RecordingModels()
        self.aio = SimpleNamespace(models=RecordingAsyncModels())


def _identity_patch(**kwargs: Any) -> Any:
    return kwargs["func"]


def test_from_genai_rejects_missing_sdk_and_wrong_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(genai_client, "Client", None)
    with pytest.raises(ClientError, match="google-genai is not installed"):
        genai_client.from_genai(cast(Client, object()))

    monkeypatch.setattr(genai_client, "Client", RecordingClient)
    with pytest.raises(ClientError, match="Got: object"):
        genai_client.from_genai(cast(Client, object()))


def test_from_genai_rejects_an_unregistered_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(genai_client, "Client", RecordingClient)

    with pytest.raises(ModeError) as error:
        genai_client.from_genai(cast(Client, RecordingClient()), mode=Mode.JSON_SCHEMA)

    assert error.value.provider == Provider.GENAI.value
    assert error.value.mode == Mode.JSON_SCHEMA.value
    assert Mode.TOOLS.value in error.value.valid_modes
    assert Mode.JSON.value in error.value.valid_modes


def test_from_genai_sync_wrapper_routes_content_and_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(genai_client, "Client", RecordingClient)
    monkeypatch.setattr(genai_client, "patch_v2", _identity_patch)
    client = RecordingClient()

    wrapped = genai_client.from_genai(
        cast(Client, client), mode=Mode.TOOLS, model="gemini-default"
    )
    content = wrapped.create_fn(contents=["one"])
    stream = list(
        wrapped.create_fn(model="gemini-override", stream=True, contents=["two"])
    )

    assert content == {
        "kind": "content",
        "model": "gemini-default",
        "contents": ["one"],
    }
    assert stream == [
        {
            "kind": "stream",
            "model": "gemini-override",
            "contents": ["two"],
        }
    ]
    assert client.models.calls == [
        ("content", {"model": "gemini-default", "contents": ["one"]}),
        ("stream", {"model": "gemini-override", "contents": ["two"]}),
    ]


@pytest.mark.asyncio
async def test_from_genai_async_wrapper_routes_content_and_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(genai_client, "Client", RecordingClient)
    monkeypatch.setattr(genai_client, "patch_v2", _identity_patch)
    client = RecordingClient()

    wrapped = genai_client.from_genai(
        cast(Client, client), mode=Mode.JSON, use_async=True, model="gemini-default"
    )
    content = await wrapped.create_fn(contents=["one"])
    stream = await wrapped.create_fn(
        model="gemini-override", stream=True, contents=["two"]
    )
    chunks = [chunk async for chunk in stream]

    assert content == {
        "kind": "content",
        "model": "gemini-default",
        "contents": ["one"],
    }
    assert chunks == [
        {
            "kind": "stream",
            "model": "gemini-override",
            "contents": ["two"],
        }
    ]
    assert client.aio.models.calls == [
        ("content", {"model": "gemini-default", "contents": ["one"]}),
        ("stream", {"model": "gemini-override", "contents": ["two"]}),
    ]


def test_genai_client_module_handles_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "google.genai" and "Client" in fromlist:
            raise ImportError("google-genai is unavailable")
        return original_import(name, globals, locals, fromlist, level)

    try:
        with monkeypatch.context() as context:
            context.setattr(builtins, "__import__", blocked_import)
            reloaded = importlib.reload(genai_client)
            assert reloaded.Client is None
            with pytest.raises(ClientError, match="google-genai is not installed"):
                reloaded.from_genai(object())
    finally:
        restored = importlib.reload(genai_client)

    assert restored.Client is not None


def test_genai_package_handles_a_failed_client_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("instructor.v2.providers.genai")
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if (
            level == 1
            and name == "client"
            and globals is not None
            and globals.get("__package__") == "instructor.v2.providers.genai"
        ):
            raise ImportError("client dependency is unavailable")
        return original_import(name, globals, locals, fromlist, level)

    try:
        with monkeypatch.context() as context:
            context.setattr(builtins, "__import__", blocked_import)
            assert importlib.reload(package).from_genai is None
    finally:
        restored = importlib.reload(package)

    assert callable(restored.from_genai)


def test_reask_tools_preserves_model_turn_and_adds_function_error() -> None:
    earlier_model_content = types.Content(
        role="model", parts=[types.Part.from_text(text="Let me check that.")]
    )
    model_content = types.Content(
        role="model",
        parts=[types.Part.from_function_call(name="Answer", args={"answer": "bad"})],
    )
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(content=None),
            SimpleNamespace(content=earlier_model_content),
            SimpleNamespace(content=model_content),
        ]
    )
    original_contents = (
        types.Content(role="user", parts=[types.Part.from_text(text="How many?")]),
    )

    result = handlers.reask_genai_tools(
        {"contents": original_contents}, response, ValueError("answer must be an int")
    )

    assert result["contents"][0] is original_contents[0]
    assert result["contents"][1] is model_content
    function_response_turn = result["contents"][2]
    assert function_response_turn.role == "user"
    assert function_response_turn.role != "tool"
    assert function_response_turn.parts[0].function_response.name == "Answer"
    assert (
        "answer must be an int"
        in function_response_turn.parts[0].function_response.response["error"]
    )
    assert len(original_contents) == 1


def test_reask_tools_without_function_call_adds_a_user_correction() -> None:
    model_content = types.Content(
        role="model", parts=[types.Part.from_text(text="The answer is many.")]
    )
    response = SimpleNamespace(candidates=[SimpleNamespace(content=model_content)])

    result = handlers.reask_genai_tools(
        {"contents": None}, response, ValueError("answer must be an int")
    )

    assert result["contents"][0] is model_content
    correction = result["contents"][1]
    assert correction.role == "user"
    assert "answer must be an int" in correction.parts[0].text
    assert "Recall the function correctly" in correction.parts[0].text


def test_parse_genai_tools_ignores_thought_parts_and_validates_function_call() -> None:
    completion = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="Let me think", thought=True),
                        types.Part.from_function_call(
                            name="Answer", args={"answer": 7}
                        ),
                    ],
                )
            )
        ]
    )

    parsed = handlers.parse_genai_tools(Answer, completion, strict=True)
    handled = handlers.GenAIToolsHandler(mode=Mode.TOOLS).parse_response(
        completion, Answer, strict=True
    )

    assert parsed == Answer(answer=7)
    assert handled == Answer(answer=7)
    assert cast(Any, handled)._raw_response is completion


class MissingTextChunk:
    def __init__(
        self,
        fallback_text: str | None,
        text_error: type[Exception] = ValueError,
    ) -> None:
        self.candidates = [
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=fallback_text)])
            )
        ]
        self.text_error = text_error

    @property
    def text(self) -> str:
        raise self.text_error("text accessor failed")


def test_streaming_extractors_handle_tools_text_fallback_and_bad_chunks() -> None:
    tool_chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(args={"answer": 8})
                        )
                    ]
                )
            )
        ]
    )
    tools_handler = handlers.GenAIToolsHandler(mode=Mode.TOOLS)
    json_handler = handlers.GenAIStructuredOutputsHandler(mode=Mode.JSON)

    assert list(tools_handler.extract_streaming_json([object(), tool_chunk])) == [
        json.dumps({"answer": 8})
    ]
    assert list(
        json_handler.extract_streaming_json(
            [object(), SimpleNamespace(text='{"answer": 9}'), MissingTextChunk("tail")]
        )
    ) == ['{"answer": 9}', "tail"]
    with pytest.raises(ValueError, match="text accessor failed"):
        list(json_handler.extract_streaming_json([MissingTextChunk(None)]))
    with pytest.raises(RuntimeError, match="text accessor failed"):
        list(
            json_handler.extract_streaming_json(
                [MissingTextChunk("must not mask an internal error", RuntimeError)]
            )
        )


@pytest.mark.asyncio
async def test_async_streaming_extractors_handle_tools_text_fallback_and_bad_chunks() -> (
    None
):
    tool_chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(args={"answer": 10})
                        )
                    ]
                )
            )
        ]
    )
    tools_handler = handlers.GenAIToolsHandler(mode=Mode.TOOLS)
    json_handler = handlers.GenAIStructuredOutputsHandler(mode=Mode.JSON)

    tool_chunks = [
        item
        async for item in tools_handler.extract_streaming_json_async(
            async_items([object(), tool_chunk])
        )
    ]
    json_chunks = [
        item
        async for item in json_handler.extract_streaming_json_async(
            async_items(
                [
                    object(),
                    SimpleNamespace(text='{"answer": 11}'),
                    MissingTextChunk("tail"),
                ]
            )
        )
    ]

    assert tool_chunks == [json.dumps({"answer": 10})]
    assert json_chunks == ['{"answer": 11}', "tail"]
    with pytest.raises(ValueError, match="text accessor failed"):
        [
            item
            async for item in json_handler.extract_streaming_json_async(
                async_items([MissingTextChunk(None)])
            )
        ]
    with pytest.raises(RuntimeError, match="text accessor failed"):
        [
            item
            async for item in json_handler.extract_streaming_json_async(
                async_items(
                    [MissingTextChunk("must not mask an internal error", RuntimeError)]
                )
            )
        ]


class StreamIterable(BaseModel, IterableBase):
    tasks: list[Answer]
    task_type: ClassVar[type[BaseModel] | None] = Answer


def test_handler_base_request_and_response_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = handlers.GenAIHandlerBase(mode=Mode.JSON)
    kwargs = {"keep": True}

    assert base._extract_system_instruction({}) is None
    assert base._wrap_streaming_model(None, stream=True) is None
    wrapped_model = base._wrap_streaming_model(Answer, stream=True)
    assert wrapped_model is not None
    assert issubclass(wrapped_model, PartialBase)
    assert base.handle_reask(kwargs, None, ValueError("bad")) == kwargs
    assert base.handle_reask(kwargs, None, ValueError("bad")) is not kwargs
    assert base.parse_response("raw", None) == "raw"
    with pytest.raises(NotImplementedError):
        base.prepare_request(Answer, {})

    iterable_stream = [SimpleNamespace(text='[{"answer": 1}, {"answer": 2}]')]
    partial_stream = [SimpleNamespace(text='{"answer":'), SimpleNamespace(text=" 3}")]
    iterable = base.parse_response(iterable_stream, StreamIterable, stream=True)
    partial = base.parse_response(partial_stream, Partial[Answer], stream=True)
    assert list(iterable) == [Answer(answer=1), Answer(answer=2)]
    assert isinstance(partial, list)
    assert partial[-1] == Answer(answer=3)

    iterable_result = StreamIterable(tasks=[Answer(answer=1), Answer(answer=2)])
    monkeypatch.setattr(
        handlers,
        "parse_genai_structured_outputs",
        lambda *_args, **_kwargs: iterable_result,
    )
    assert base.parse_response(object(), Answer) == [
        Answer(answer=1),
        Answer(answer=2),
    ]

    parallel_result = [Answer(answer=3)]
    monkeypatch.setattr(
        handlers,
        "parse_genai_structured_outputs",
        lambda *_args, **_kwargs: parallel_result,
    )
    parallel_model = cast(type[BaseModel], ParallelBase(Answer))
    assert base.parse_response(object(), parallel_model) is parallel_result

    class AnswerAdapter(AdapterBase):
        content: int

    monkeypatch.setattr(
        handlers,
        "parse_genai_structured_outputs",
        lambda *_args, **_kwargs: AnswerAdapter(content=4),
    )
    assert base.parse_response(object(), AnswerAdapter) == 4


@pytest.mark.asyncio
async def test_handler_base_async_streaming_response_path() -> None:
    handler = handlers.GenAIStructuredOutputsHandler(mode=Mode.JSON)
    stream = async_items([SimpleNamespace(text='[{"answer": 1}, {"answer": 2}]')])

    result = cast(
        AsyncGenerator[BaseModel, None],
        handler.parse_response(stream, StreamIterable, stream=True, is_async=True),
    )

    assert [item async for item in result] == [Answer(answer=1), Answer(answer=2)]


def test_prepare_request_without_model_keeps_explicit_system_instruction() -> None:
    handler = handlers.GenAIToolsHandler(mode=Mode.TOOLS)

    model, request = handler.prepare_request(
        None,
        {
            "system": "Keep the answer short.",
            "messages": [{"role": "user", "content": "How many?"}],
            "temperature": 0.2,
        },
    )

    assert model is None
    assert request["config"].system_instruction == "Keep the answer short."
    assert request["contents"][0].parts[0].text == "How many?"
    assert "system" not in request
    assert "temperature" not in request


@pytest.mark.parametrize(
    ("handler", "expected_mode"),
    [
        (handlers.GenAIToolsHandler(mode=Mode.TOOLS), "tools"),
        (handlers.GenAIStructuredOutputsHandler(mode=Mode.JSON), "json"),
    ],
)
def test_prepare_request_merges_openai_generation_settings(
    handler: handlers.GenAIHandlerBase, expected_mode: str
) -> None:
    original = {
        "messages": [
            {"role": "system", "content": "Return a number."},
            {"role": "user", "content": "How many?"},
        ],
        "max_tokens": 32,
        "temperature": 0.25,
        "top_p": 0.8,
        "n": 1,
        "stop": ["END"],
        "seed": 2,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
    }

    prepared_model, request = handler.prepare_request(Answer, original)

    assert prepared_model is not None
    assert prepared_model.__name__ == "Answer"
    assert prepared_model.model_validate({"answer": 4}).model_dump() == {"answer": 4}
    assert "messages" not in request
    assert len(request["contents"]) == 1
    assert request["contents"][0].role == "user"
    assert request["contents"][0].parts[0].text == "How many?"
    config = request["config"]
    assert config.max_output_tokens == 32
    assert config.temperature == 0.25
    assert config.top_p == 0.8
    assert config.candidate_count == 1
    assert config.stop_sequences == ["END"]
    assert config.seed == 2
    assert config.presence_penalty == 0.1
    assert config.frequency_penalty == 0.2
    assert config.system_instruction.strip() == "Return a number."
    if expected_mode == "tools":
        assert config.tools[0].function_declarations[0].name == "Answer"
        assert config.tool_config.function_calling_config.allowed_function_names == [
            "Answer"
        ]
    else:
        assert config.response_mime_type == "application/json"
        assert config.response_schema is prepared_model
    assert original["max_tokens"] == 32
    assert len(original["messages"]) == 2


def test_multimodal_types_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "google.genai" and "types" in fromlist:
            raise ImportError("google-genai is unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="google-genai package is required"):
        multimodal._types()


def test_multimodal_encodes_gcs_image_and_audio() -> None:
    image = SimpleNamespace(
        source="gs://bucket/image.png", media_type="image/png", data=b"png-bytes"
    )
    audio = SimpleNamespace(
        source="recording.wav", media_type="audio/wav", data="d2F2LWJ5dGVz"
    )

    image_part = multimodal.image_to_genai(image)
    audio_part = multimodal.audio_to_genai(audio)

    assert image_part.inline_data.data == b"png-bytes"
    assert image_part.inline_data.mime_type == "image/png"
    assert audio_part.inline_data.data == b"wav-bytes"
    assert audio_part.inline_data.mime_type == "audio/wav"


def _install_pdf_upload_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    active_after_poll: bool,
    pending_name: str | None = "files/pending",
) -> tuple[list[tuple[str, str]], list[int]]:
    pending = SimpleNamespace(
        state=types.FileState.PROCESSING,
        name=pending_name,
        uri="gs://bucket/pending.pdf",
        mime_type="application/pdf",
    )
    polled = SimpleNamespace(
        state=(
            types.FileState.ACTIVE if active_after_poll else types.FileState.PROCESSING
        ),
        name="files/pending",
        uri="gs://bucket/ready.pdf" if active_after_poll else pending.uri,
        mime_type="application/pdf",
    )
    calls: list[tuple[str, str]] = []

    class Files:
        def upload(self, *, file: str) -> SimpleNamespace:
            calls.append(("upload", file))
            return pending

        def get(self, *, name: str) -> SimpleNamespace:
            calls.append(("get", name))
            return polled

    class Client:
        def __init__(self) -> None:
            self.files = Files()

    sleeps: list[int] = []
    monkeypatch.setattr("google.genai.Client", Client)
    monkeypatch.setattr("time.sleep", sleeps.append)
    return calls, sleeps


@pytest.mark.parametrize(
    ("max_retries", "expected_calls", "expected_sleeps"),
    [
        (0, [("upload", "/tmp/pending.pdf")], []),
        (
            1,
            [("upload", "/tmp/pending.pdf"), ("get", "files/pending")],
            [3],
        ),
    ],
)
def test_upload_new_pdf_file_stops_at_the_retry_limit(
    max_retries: int,
    expected_calls: list[tuple[str, str]],
    expected_sleeps: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, sleeps = _install_pdf_upload_client(monkeypatch, active_after_poll=False)

    with pytest.raises(Exception, match="Max retries reached"):
        multimodal.upload_new_pdf_file(
            SimpleNamespace,
            "/tmp/pending.pdf",
            retry_delay=3,
            max_retries=max_retries,
        )

    assert calls == expected_calls
    assert sleeps == expected_sleeps


def test_upload_new_pdf_file_recovers_on_the_last_allowed_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, sleeps = _install_pdf_upload_client(monkeypatch, active_after_poll=True)

    result = multimodal.upload_new_pdf_file(
        SimpleNamespace, "/tmp/pending.pdf", retry_delay=3, max_retries=1
    )

    assert result == SimpleNamespace(
        source="gs://bucket/ready.pdf", media_type="application/pdf", data=None
    )
    assert calls == [("upload", "/tmp/pending.pdf"), ("get", "files/pending")]
    assert sleeps == [3]


def test_upload_new_pdf_file_rejects_a_pending_file_without_a_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, sleeps = _install_pdf_upload_client(
        monkeypatch, active_after_poll=False, pending_name=None
    )

    with pytest.raises(ValueError, match="Cannot poll a pending GenAI file"):
        multimodal.upload_new_pdf_file(
            SimpleNamespace, "/tmp/pending.pdf", retry_delay=3, max_retries=1
        )

    assert calls == [("upload", "/tmp/pending.pdf")]
    assert sleeps == []


def test_extract_multimodal_content_preserves_uploaded_files() -> None:
    uploaded = types.File(
        name="files/ready", uri="gs://bucket/ready.pdf", mime_type="application/pdf"
    )

    result = multimodal.extract_multimodal_content([uploaded])

    assert result == [uploaded]
    assert result[0] is uploaded
