from __future__ import annotations

import asyncio
import importlib
import json
import threading
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from instructor import Mode
from instructor.v2.core.errors import InstructorRetryException, ResponseParsingError
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.dsl.parallel import VertexAIParallelModel
from tests.coverage._openai import chat_completion


class Contract(BaseModel):
    model_config = ConfigDict(title="extract_contract")
    value: bool

    @field_validator("value")
    @classmethod
    def apply_context(cls, value: bool, info: ValidationInfo) -> bool:
        return not value if (info.context or {}).get("invert") else value


class ExtraTitle(Contract):
    model_config = ConfigDict(json_schema_extra={"title": "extra_contract"})


CASES = [
    (Provider.OPENAI, Mode.TOOLS),
    (Provider.OPENAI, Mode.PARALLEL_TOOLS),
    (Provider.ANTHROPIC, Mode.TOOLS),
    (Provider.ANTHROPIC, Mode.PARALLEL_TOOLS),
    (Provider.MISTRAL, Mode.TOOLS),
    (Provider.XAI, Mode.PARALLEL_TOOLS),
    (Provider.VERTEXAI, Mode.PARALLEL_TOOLS),
]


def response_payload(
    provider: Provider, calls: list[tuple[str, Any]]
) -> dict[str, Any]:
    if provider == Provider.ANTHROPIC:
        return {
            "id": "msg_parallel",
            "type": "message",
            "role": "assistant",
            "model": "local-model",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [
                {
                    "type": "tool_use",
                    "id": f"call_{i}",
                    "name": name,
                    "input": {"value": value},
                }
                for i, (name, value) in enumerate(calls)
            ],
        }
    return {
        "id": "chatcmpl_parallel",
        "object": "chat.completion",
        "created": 0,
        "model": "local-model",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps({"value": value}),
                            },
                        }
                        for i, (name, value) in enumerate(calls)
                    ],
                },
            }
        ],
    }


def sdk_response(provider: Provider, calls: list[tuple[str, Any]]) -> Any:
    if provider == Provider.ANTHROPIC:
        message = pytest.importorskip("anthropic.types").Message
        return message.model_validate(response_payload(provider, calls))
    if provider == Provider.MISTRAL:
        pytest.importorskip("mistralai")
        try:
            models = importlib.import_module("mistralai.client.models")
        except ImportError:
            models = importlib.import_module("mistralai.models")
        return models.ChatCompletionResponse.model_validate(
            response_payload(provider, calls)
        )
    if provider == Provider.XAI:
        chat = pytest.importorskip("xai_sdk.chat")
        return chat.Response(xai_response(calls), 0)
    if provider == Provider.VERTEXAI:
        gm = pytest.importorskip("vertexai.generative_models")
        return gm.GenerationResponse.from_dict(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "function_call": {
                                        "name": name,
                                        "args": {"value": value},
                                    }
                                }
                                for name, value in calls
                            ],
                        }
                    }
                ]
            }
        )
    completion = pytest.importorskip("openai.types.chat").ChatCompletion
    return completion.model_validate(response_payload(provider, calls))


def parse(
    provider: Provider, mode: Mode, response: Any, model: type[BaseModel], strict: bool
) -> Any:
    response_model = cast(Any, Iterable)[model]
    if provider == Provider.VERTEXAI:
        response_model = cast(Any, VertexAIParallelModel(response_model))
    return mode_registry.get_handlers(provider, mode).response_parser(
        response=response,
        response_model=response_model,
        validation_context={"invert": True},
        strict=strict,
        stream=False,
    )


@pytest.mark.parametrize("provider,mode", CASES)
@pytest.mark.parametrize("model", [Contract, ExtraTitle])
def test_declared_names_context_and_strict_validation(
    provider: Provider, mode: Mode, model: type[BaseModel]
) -> None:
    name = (
        model.__name__
        if provider == Provider.VERTEXAI
        else model.model_json_schema()["title"]
    )
    assert [
        r.model_dump()
        for r in parse(
            provider, mode, sdk_response(provider, [(name, True)]), model, True
        )
    ] == [{"value": False}]
    assert [
        r.model_dump()
        for r in parse(
            provider, mode, sdk_response(provider, [(name, "true")]), model, False
        )
    ] == [{"value": False}]
    with pytest.raises(ValidationError):
        parse(
            provider,
            mode,
            sdk_response(provider, [(name, True), (name, "true")]),
            model,
            True,
        )


@pytest.mark.parametrize("provider,mode", CASES)
@pytest.mark.parametrize("unknown_first", [False, True])
def test_unknown_call_rejected_before_iterator_escapes(
    provider: Provider, mode: Mode, unknown_first: bool
) -> None:
    name = Contract.__name__ if provider == Provider.VERTEXAI else "extract_contract"
    calls = [(name, True), ("Unknown", True)]
    if unknown_first:
        calls.reverse()
    response = sdk_response(provider, calls)
    with pytest.raises(ResponseParsingError, match="Unknown") as error:
        parse(provider, mode, response, Contract, True)
    assert error.value.raw_response is response
    assert name in str(error.value)


def test_openai_tools_parallel_missing_tool_calls_is_retryable() -> None:
    response = chat_completion(content="Just text, no tools.")
    with pytest.raises(ResponseParsingError, match="No tool calls") as error:
        parse(Provider.OPENAI, Mode.TOOLS, response, Contract, True)
    assert error.value.raw_response is response


@pytest.fixture
def local_http() -> Iterator[tuple[str, deque[dict[str, Any]], list[dict[str, Any]]]]:
    replies: deque[dict[str, Any]] = deque()
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            payload = json.dumps(replies.popleft()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", replies, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


HTTP_CASES = [
    case
    for case in CASES
    if case[0] in {Provider.OPENAI, Provider.ANTHROPIC, Provider.MISTRAL}
]


def http_client(
    provider: Provider, mode: Mode, url: str, asynchronous: bool
) -> tuple[Any, Any]:
    import httpx

    if provider == Provider.OPENAI:
        from openai import AsyncOpenAI, OpenAI
        from instructor.v2.providers.openai.client import from_openai

        cls = AsyncOpenAI if asynchronous else OpenAI
        sdk = cls(api_key="local-only", base_url=url, max_retries=0)
        return from_openai(sdk, mode=mode), sdk
    if provider == Provider.ANTHROPIC:
        from anthropic import Anthropic, AsyncAnthropic
        from instructor.v2.providers.anthropic.client import from_anthropic

        cls = AsyncAnthropic if asynchronous else Anthropic
        sdk = cls(api_key="local-only", base_url=url, max_retries=0)
        return from_anthropic(sdk, mode=mode), sdk
    from instructor.v2.providers.mistral.client import Mistral, from_mistral

    assert Mistral is not None
    sdk = Mistral(
        api_key="local-only",
        server_url=url,
        client=httpx.Client(trust_env=False),
        async_client=httpx.AsyncClient(trust_env=False),
    )
    return from_mistral(sdk, mode=mode, use_async=asynchronous), sdk


@pytest.mark.parametrize("provider,mode", HTTP_CASES)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("exhaust", [False, True])
def test_unknown_call_retries_inside_sdk_http_boundary(
    local_http: Any, provider: Provider, mode: Mode, asynchronous: bool, exhaust: bool
) -> None:
    url, replies, requests = local_http
    bad = response_payload(provider, [("extract_contract", True), ("Unknown", True)])
    replies.extend(
        [
            bad,
            bad
            if exhaust
            else response_payload(provider, [("extract_contract", True)]),
        ]
    )
    wrapped, sdk = http_client(provider, mode, url, asynchronous)
    kwargs = dict(
        response_model=Iterable[Contract],
        messages=[{"role": "user", "content": "Extract"}],
        model="local-model",
        max_tokens=64,
        max_retries=1,
        context={"invert": True},
        strict=True,
    )

    def call() -> Any:
        if mode == Mode.PARALLEL_TOOLS:
            return wrapped.create(**kwargs)
        # TOOLS public clients wrap Iterable[T] in one tasks tool. Exercise the
        # parallel-compatible handler with the real SDK and shared retry loop.
        from instructor.v2.core.retry import retry_async_v2, retry_sync_v2

        handlers = mode_registry.get_handlers(provider, mode)
        prepared, request = handlers.request_handler(
            response_model=cast(Any, Iterable[Contract]),
            kwargs={
                "messages": kwargs["messages"],
                "model": "local-model",
                "max_tokens": 64,
            },
        )
        if provider == Provider.OPENAI:
            endpoint = sdk.chat.completions.create
        elif provider == Provider.ANTHROPIC:
            endpoint = sdk.messages.create
        else:
            endpoint = sdk.chat.complete_async if asynchronous else sdk.chat.complete
        retry = retry_async_v2 if asynchronous else retry_sync_v2
        return retry(
            func=endpoint,
            response_model=prepared,
            provider=provider,
            mode=mode,
            context={"invert": True},
            max_retries=1,
            args=(),
            kwargs=request,
            strict=True,
        )

    async def run_async() -> Any:
        try:
            return await call()
        finally:
            close = getattr(sdk, "close", None)
            if close is not None:
                await close()
            else:
                await sdk.sdk_configuration.async_client.aclose()
                sdk.sdk_configuration.client.close()

    try:
        if exhaust:
            with pytest.raises(InstructorRetryException) as error:
                asyncio.run(run_async()) if asynchronous else call()
            assert error.value.failed_attempts is not None
            assert len(error.value.failed_attempts) == 2
            assert all(
                isinstance(a.exception, ResponseParsingError)
                for a in error.value.failed_attempts
            )
        else:
            iterator = asyncio.run(run_async()) if asynchronous else call()
            assert [r.model_dump() for r in iterator] == [{"value": False}]
    finally:
        if not asynchronous:
            close = getattr(sdk, "close", None)
            if close is not None:
                close()
            else:
                sdk.sdk_configuration.client.close()
                asyncio.run(sdk.sdk_configuration.async_client.aclose())
    assert len(requests) == 2
    tools = requests[0]["tools"]
    assert (
        tools[0]["name"]
        if provider == Provider.ANTHROPIC
        else tools[0]["function"]["name"]
    ) == "extract_contract"
    assert len(requests[1]["messages"]) > len(requests[0]["messages"])


def xai_response(calls: list[tuple[str, Any]]) -> Any:
    pb = pytest.importorskip("xai_sdk.proto").chat_pb2
    return pb.GetChatCompletionResponse(
        id="local_parallel",
        model="local-model",
        choices=[
            pb.Choice(
                index=0,
                message=pb.CompletionMessage(
                    tool_calls=[
                        pb.ToolCall(
                            id=f"call_{i}",
                            function={
                                "name": name,
                                "arguments": json.dumps({"value": value}),
                            },
                        )
                        for i, (name, value) in enumerate(calls)
                    ]
                ),
            )
        ],
    )


@pytest.fixture
def local_xai() -> Iterator[tuple[str, deque[Any], list[Any]]]:
    grpc = pytest.importorskip("grpc")
    chat_grpc = pytest.importorskip("xai_sdk.proto").chat_pb2_grpc
    replies: deque[Any] = deque()
    requests: list[Any] = []

    class Service(chat_grpc.ChatServicer):
        def GetCompletion(self, request: Any, _context: Any) -> Any:
            requests.append(request)
            return replies.popleft()

    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    chat_grpc.add_ChatServicer_to_server(Service(), server)
    port = server.add_secure_port("127.0.0.1:0", grpc.local_server_credentials())
    server.start()
    try:
        yield f"localhost:{port}", replies, requests
    finally:
        server.stop(0).wait()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("exhaust", [False, True])
def test_native_xai_parallel_retries_names_and_preserves_context(
    local_xai: Any, asynchronous: bool, exhaust: bool
) -> None:
    host, replies, requests = local_xai
    bad = xai_response([("extract_contract", True), ("Unknown", True)])
    replies.extend(
        [bad, bad if exhaust else xai_response([("extract_contract", True)])]
    )
    from instructor.v2.providers.xai.client import from_xai

    module = pytest.importorskip(
        "xai_sdk.aio.client" if asynchronous else "xai_sdk.sync.client"
    )

    def call(sdk: Any) -> Any:
        return from_xai(sdk, mode=Mode.PARALLEL_TOOLS).create(
            response_model=Iterable[Contract],
            messages=[{"role": "user", "content": "Extract"}],
            model="local-model",
            max_retries=1,
            context={"invert": True},
            strict=True,
        )

    async def run_async() -> Any:
        return await call(module.Client(api_key="local-only", api_host=host, timeout=5))

    if exhaust:
        with pytest.raises(InstructorRetryException) as error:
            asyncio.run(run_async()) if asynchronous else call(
                module.Client(api_key="local-only", api_host=host, timeout=5)
            )
        assert error.value.failed_attempts is not None
        assert len(error.value.failed_attempts) == 2
        assert all(
            isinstance(a.exception, ResponseParsingError)
            for a in error.value.failed_attempts
        )
    else:
        iterator = (
            asyncio.run(run_async())
            if asynchronous
            else call(module.Client(api_key="local-only", api_host=host, timeout=5))
        )
        results = list(iterator)
        assert [r.model_dump() for r in results] == [{"value": False}]
        raw = results[0]._raw_response
        assert raw.id == "local_parallel"
        assert [c.function.name for c in raw.tool_calls] == ["extract_contract"]
    assert len(requests) == 2
    assert [t.function.name for t in requests[0].tools] == ["extract_contract"]
    assert len(requests[1].messages) > len(requests[0].messages)
