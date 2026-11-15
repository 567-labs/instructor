from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, Iterable
from copy import copy
from typing import Any, cast, get_args, get_origin

from tests import openai_http as httpx
import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, PrivateAttr
from typing_extensions import get_overloads

from instructor.v2.core import client as core_client
from instructor.v2.core.client import (
    AsyncInstructor,
    AsyncResponse,
    Instructor,
    Response,
)
from instructor.v2.core.hooks import HookName, Hooks
from instructor.v2.core.mode import Mode
from tests._streams import async_items


class User(BaseModel):
    name: str
    _raw_response: Any = PrivateAttr(default=None)


MESSAGES: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "Return a user"}
]


def hooks_with_handler(label: str, events: list[str]) -> Hooks:
    hooks = Hooks()
    hooks.on(HookName.COMPLETION_RESPONSE, lambda _response: events.append(label))
    return hooks


def assert_combined_hooks(hooks: Hooks, events: list[str]) -> None:
    hooks.emit_completion_response(object())
    assert events[-2:] == ["client", "call"]


def test_response_normalizes_input_alias_and_rejects_ambiguous_messages() -> None:
    kwargs: dict[str, Any] = {"input": "hello", "temperature": 0}

    assert Response._normalize_messages(None, kwargs) == [
        {"role": "user", "content": "hello"}
    ]
    assert kwargs == {"temperature": 0}
    assert Response._normalize_messages(MESSAGES, {}) is MESSAGES

    with pytest.raises(TypeError, match="Either 'messages' or 'input'"):
        Response._normalize_messages(None, {})
    with pytest.raises(TypeError, match="Pass only one of 'messages' or 'input'"):
        Response._normalize_messages(MESSAGES, {"input": "duplicate"})


def test_sync_client_forwards_defaults_aliases_and_per_call_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        Mode,
        "warn_mode_functions_deprecation",
        lambda: warnings.append("functions"),
    )
    events: list[str] = []
    client_hooks = hooks_with_handler("client", events)
    call_hooks = hooks_with_handler("call", events)
    received: list[dict[str, Any]] = []
    underlying = type("UnderlyingClient", (), {"account": "local"})()

    def create(**kwargs: Any) -> User:
        received.append(kwargs)
        return User(name="Ada")

    client = Instructor(
        client=underlying,
        create=create,
        mode=Mode.FUNCTIONS,
        hooks=client_hooks,
        model="default-model",
        temperature=0.1,
    )

    result = client.create(
        response_model=User,
        messages=MESSAGES,
        hooks=call_hooks,
        model="call-model",
    )

    assert result == User(name="Ada")
    assert warnings == ["functions"]
    assert received[0]["model"] == "call-model"
    assert received[0]["temperature"] == 0.1
    assert received[0]["messages"] is MESSAGES
    assert received[0]["response_model"] is User
    assert_combined_hooks(received[0]["hooks"], events)
    assert client.chat.completions.messages is client
    assert client.account == "local"
    assert Instructor.__getattr__(client, "create").__self__ is client


@pytest.mark.parametrize("client_class", [Instructor, AsyncInstructor])
@pytest.mark.parametrize("underlying", [None, object()])
def test_shallow_copy_preserves_client_state(
    client_class: type[Instructor], underlying: Any
) -> None:
    def create(**_kwargs: Any) -> User:
        return User(name="Ada")

    client = client_class(
        client=underlying,
        create=create,
        model="default-model",
    )

    copied = copy(client)

    assert copied is not client
    assert type(copied) is client_class
    assert copied.client is underlying
    assert copied.create_fn is create
    assert copied.kwargs is client.kwargs
    assert copied.hooks is client.hooks
    assert copied.chat.completions.messages is copied
    with pytest.raises(AttributeError):
        _ = copied.missing_attribute


sync_and_async = pytest.mark.parametrize(
    "is_async", [False, True], ids=["sync", "async"]
)


def recording_create(
    received: list[dict[str, Any]], raw: object, is_async: bool, name: str = "Ada"
) -> Any:
    """Create function returning a model, or a stream when `stream=True`."""

    def respond(kwargs: dict[str, Any]) -> Any:
        received.append(kwargs)
        if kwargs.get("stream"):
            items = (
                [User(name="Ada"), User(name="Grace")]
                if get_origin(kwargs["response_model"]) is Iterable
                else [User.model_construct(name=None), User(name="Ada")]
            )
            return async_items(items) if is_async else iter(items)
        result = User(name=name)
        result._raw_response = raw
        return result

    if is_async:

        async def acreate(**kwargs: Any) -> Any:
            return respond(kwargs)

        return acreate
    return lambda **kwargs: respond(kwargs)


async def resolve(value: Any, is_async: bool) -> Any:
    return await value if is_async else value


async def drain(stream: Any, is_async: bool) -> list[Any]:
    if not is_async:
        return list(stream)
    if inspect.isawaitable(stream):
        stream = await stream
    return [item async for item in stream]


@pytest.mark.asyncio
@sync_and_async
@pytest.mark.parametrize(
    "mode", [Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS]
)
async def test_shallow_copy_rebinds_response_facade(mode: Mode, is_async: bool) -> None:
    received: list[dict[str, Any]] = []
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, request=request)
    )
    sdk: Any = openai.AsyncOpenAI if is_async else openai.OpenAI
    http: Any = httpx.AsyncClient if is_async else httpx.Client
    native_client = sdk(
        api_key="offline-key",
        base_url="https://offline.invalid/v1",
        http_client=http(transport=transport),
    )
    client_class: Any = AsyncInstructor if is_async else Instructor
    facade = AsyncResponse if is_async else Response
    client = client_class(
        client=native_client,
        create=recording_create(received, None, is_async, "Original"),
        mode=mode,
        model="original-model",
    )
    copied = copy(client)
    copied.create_fn = recording_create(received, None, is_async, "Copied")
    copied.kwargs = {"model": "copied-model"}
    copied.hooks = Hooks()

    assert isinstance(copied.responses, facade)
    assert isinstance(client.responses, facade)
    for target, name in [(copied, "Copied"), (client, "Original")]:
        result = target.responses.create(input="Return a user", response_model=User)
        assert await resolve(result, is_async) == User(name=name)
    assert copied.responses is not client.responses
    assert copied.responses.client is copied
    assert client.responses.client is client
    assert copied.client is native_client
    assert [call["model"] for call in received] == ["copied-model", "original-model"]
    assert received[0]["hooks"] is copied.hooks
    assert received[1]["hooks"] is client.hooks
    await resolve(native_client.close(), is_async)


@pytest.mark.asyncio
@sync_and_async
async def test_streaming_and_completion_helpers_preserve_models_and_hooks(
    is_async: bool,
) -> None:
    events: list[str] = []
    call_hooks = hooks_with_handler("call", events)
    received: list[dict[str, Any]] = []
    raw = object()
    client_class: Any = AsyncInstructor if is_async else Instructor
    client = client_class(
        client=None,
        create=recording_create(received, raw, is_async),
        hooks=hooks_with_handler("client", events),
        model="default-model",
    )
    options: dict[str, Any] = {"response_model": User, "messages": MESSAGES}

    partial = await drain(
        client.create_partial(**options, hooks=call_hooks, strict=False), is_async
    )
    iterable = await drain(
        client.create_iterable(**options, hooks=call_hooks), is_async
    )
    parsed, completion = await resolve(
        client.create_with_completion(**options, hooks=call_hooks), is_async
    )

    assert partial[0].name is None
    assert partial[1:] == [User(name="Ada")]
    assert received[0]["response_model"]._original_model is User
    assert received[0]["stream"] is True
    assert received[0]["strict"] is False
    assert iterable == [User(name="Ada"), User(name="Grace")]
    assert get_origin(received[1]["response_model"]) is Iterable
    assert get_args(received[1]["response_model"]) == (User,)
    assert received[1]["stream"] is True
    assert parsed.model_dump() == {"name": "Ada"}
    assert completion is raw
    assert all(call["model"] == "default-model" for call in received)
    for call in received:
        assert_combined_hooks(call["hooks"], events)


@pytest.mark.asyncio
@sync_and_async
async def test_response_facade_normalizes_input_and_forwards_each_helper(
    is_async: bool,
) -> None:
    received: list[dict[str, Any]] = []
    raw = object()
    client_class: Any = AsyncInstructor if is_async else Instructor
    client = client_class(
        client=None,
        create=recording_create(received, raw, is_async),
        model="offline-model",
    )
    response: Any = (AsyncResponse if is_async else Response)(client)

    parsed = await resolve(
        response.create(
            input="Return Ada",
            response_model=User,
            max_retries=1,
            strict=False,
            context={"source": "test"},
        ),
        is_async,
    )
    with_completion, completion = await resolve(
        response.create_with_completion(
            input="Return Ada and the raw response", response_model=User, max_retries=2
        ),
        is_async,
    )
    iterable = await drain(
        response.create_iterable(
            input="Return two users", response_model=User, max_retries=3
        ),
        is_async,
    )
    partial = await drain(
        response.create_partial(
            input="Stream two users", response_model=User, max_retries=4
        ),
        is_async,
    )

    assert parsed.model_dump() == {"name": "Ada"}
    assert with_completion.model_dump() == {"name": "Ada"}
    assert completion is raw
    assert iterable == [User(name="Ada"), User(name="Grace")]
    assert [item.name for item in partial] == [None, "Ada"]
    assert [call["messages"][0]["content"] for call in received] == [
        "Return Ada",
        "Return Ada and the raw response",
        "Return two users",
        "Stream two users",
    ]
    assert [call["max_retries"] for call in received] == [1, 2, 3, 4]
    assert received[0]["strict"] is False
    assert received[0]["context"] == {"source": "test"}
    assert received[2]["stream"] is True
    assert received[3]["stream"] is True
    assert all(call["hooks"] is client.hooks for call in received)
    assert all(call["model"] == "offline-model" for call in received)


@pytest.mark.asyncio
async def test_async_client_forwards_defaults_and_dispatches_iterable_response() -> (
    None
):
    events: list[str] = []
    client_hooks = hooks_with_handler("client", events)
    call_hooks = hooks_with_handler("call", events)
    received: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        received.append(kwargs)
        if kwargs.get("stream"):

            async def stream() -> Any:
                yield User(name="Ada")
                yield User(name="Grace")

            return stream()
        return User(name="Ada")

    client = AsyncInstructor(
        client=None,
        create=create,
        hooks=client_hooks,
        model="default-model",
    )

    parsed = await client.create(
        response_model=User,
        messages=MESSAGES,
        hooks=call_hooks,
        temperature=0,
    )
    stream = await client.create(
        response_model=Iterable[User],
        messages=MESSAGES,
        hooks=call_hooks,
    )
    iterable = [item async for item in cast(AsyncIterable[User], stream)]

    assert parsed.model_dump() == {"name": "Ada"}
    assert iterable == [User(name="Ada"), User(name="Grace")]
    assert received[0]["model"] == "default-model"
    assert received[0]["temperature"] == 0
    assert get_origin(received[1]["response_model"]) is Iterable
    assert get_args(received[1]["response_model"]) == (User,)
    assert received[1]["stream"] is True
    for call in received:
        assert_combined_hooks(call["hooks"], events)


def test_client_hook_helpers_register_remove_and_clear_handlers() -> None:
    events: list[str] = []

    def on_response(_response: Any) -> None:
        events.append("response")

    def on_error(_error: Any) -> None:
        events.append("error")

    client = Instructor(client=None, create=lambda **_kwargs: User(name="Ada"))

    client.on("completion:response", on_response)
    client.hooks.emit_completion_response(object())
    client.off("completion:response", on_response)
    client.hooks.emit_completion_response(object())
    assert events == ["response"]

    client.on(HookName.COMPLETION_RESPONSE, on_response)
    client.on(HookName.COMPLETION_ERROR, on_error)
    client.clear(HookName.COMPLETION_RESPONSE)
    client.hooks.emit_completion_response(object())
    client.hooks.emit_completion_error(RuntimeError("offline failure"))
    assert events == ["response", "error"]

    client.clear()
    client.hooks.emit_completion_error(RuntimeError("another failure"))
    assert events == ["response", "error"]


def test_openai_compat_wrapper_delegates_and_keeps_its_sync_async_overloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from instructor.v2.providers.openai import client as openai_client

    received: list[dict[str, Any]] = []
    expected = object()
    source_client = cast(openai.OpenAI, object())

    monkeypatch.setattr(
        openai_client,
        "from_openai",
        lambda **kwargs: received.append(kwargs) or expected,
    )

    assert (
        core_client.from_openai(
            source_client,
            mode=Mode.JSON,
            model="local-model",
        )
        is expected
    )
    assert received == [
        {"client": source_client, "mode": Mode.JSON, "model": "local-model"}
    ]

    overloads = get_overloads(core_client.from_openai)
    assert len(overloads) == 2
    assert [
        tuple(inspect.signature(overload).parameters) for overload in overloads
    ] == [
        ("client", "mode", "kwargs"),
        ("client", "mode", "kwargs"),
    ]


@pytest.mark.parametrize("async_client", [True, False, None])
def test_litellm_compat_wrapper_preserves_explicit_and_inferred_client_modes(
    monkeypatch: pytest.MonkeyPatch,
    async_client: bool | None,
) -> None:
    from instructor.v2.providers.litellm import client as litellm_client

    received: list[dict[str, Any]] = []
    expected = object()

    def completion(**_kwargs: Any) -> object:
        return object()

    async def async_completion(**_kwargs: Any) -> object:
        return object()

    monkeypatch.setattr(
        litellm_client,
        "from_litellm",
        lambda **kwargs: received.append(kwargs) or expected,
    )

    if async_client is True:
        wrapped = core_client.from_litellm(
            async_completion,
            mode=Mode.JSON,
            async_client=True,
            model="local-model",
        )
    elif async_client is False:
        wrapped = core_client.from_litellm(
            completion,
            mode=Mode.JSON,
            async_client=False,
            model="local-model",
        )
    else:
        wrapped = core_client.from_litellm(
            completion,
            mode=Mode.JSON,
            model="local-model",
        )
    assert wrapped is expected
    forwarded = received.pop()
    expected_completion = async_completion if async_client is True else completion
    assert forwarded["completion"] is expected_completion
    assert forwarded["mode"] is Mode.JSON
    assert forwarded["model"] == "local-model"
    if async_client is None:
        assert "async_client" not in forwarded
    else:
        assert forwarded["async_client"] is async_client
