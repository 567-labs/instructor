from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict

import instructor.v2.core.response as response_module
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import ModeRegistry
from instructor.v2.core.response import (
    _ensure_registry_loaded,
    handle_response_model,
    is_typed_dict,
    process_response,
    process_response_async,
)
from instructor.v2.dsl.iterable import IterableBase, IterableModel
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.partial import Partial
from instructor.v2.dsl.response_list import ListResponse
from instructor.v2.dsl.simple_type import AdapterBase
from tests.coverage._openai import chat_completion, tool_call, tool_chunks
from tests.coverage._streams import async_items


class User(BaseModel):
    name: str
    age: int


class UsersEnvelope(BaseModel, IterableBase):
    tasks: list[User]

    @classmethod
    def from_response(
        cls,
        response: Any,
        *,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        mode: Mode,
    ) -> UsersEnvelope:
        assert mode is Mode.JSON
        return cls.model_validate_json(
            response.content[0].text,
            context=validation_context,
            strict=strict,
        )


class IntegerAdapter(AdapterBase):
    content: int


def _install_parser(
    monkeypatch: pytest.MonkeyPatch,
    parser: Callable[..., Any],
) -> None:
    registry = ModeRegistry()

    def request_handler(
        response_model: Any, kwargs: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        return response_model, kwargs

    def reask_handler(
        kwargs: dict[str, Any], response: Any, exception: Exception
    ) -> dict[str, Any]:
        del response, exception
        return kwargs

    registry.register(
        Provider.UNKNOWN,
        Mode.JSON,
        request_handler=request_handler,
        reask_handler=reask_handler,
        response_parser=parser,
    )
    monkeypatch.setattr(response_module, "mode_registry", registry)


def test_registry_load_failure_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []

    def unavailable(name: str) -> None:
        attempted.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", unavailable)

    assert _ensure_registry_loaded() is None
    assert attempted == ["instructor.v2"]


@pytest.mark.asyncio
async def test_raw_response_is_returned_unchanged_sync_and_async() -> None:
    raw = chat_completion(content='{"name": "Ada", "age": 36}')

    assert process_response(raw, response_model=None, stream=False) is raw
    assert await process_response_async(raw, response_model=None) is raw


@pytest.mark.asyncio
async def test_standard_tool_response_attaches_raw_completion_sync_and_async() -> None:
    raw = chat_completion(tool_calls=[tool_call("User", {"name": "Ada", "age": 36})])

    sync_result = process_response(
        raw, response_model=User, stream=False, mode=Mode.TOOLS
    )
    async_result = await process_response_async(
        raw, response_model=User, stream=False, mode=Mode.TOOLS
    )

    assert sync_result == User(name="Ada", age=36)
    assert async_result == User(name="Ada", age=36)
    assert sync_result._raw_response is raw
    assert async_result._raw_response is raw


@pytest.mark.asyncio
async def test_invalid_tool_arguments_raise_validation_error_sync_and_async() -> None:
    raw = chat_completion(tool_calls=[tool_call("User", '{"name": "Ada"}')])

    with pytest.raises(ValidationError, match="age"):
        process_response(raw, response_model=User, stream=False, mode=Mode.TOOLS)

    with pytest.raises(ValidationError, match="age"):
        await process_response_async(
            raw, response_model=User, stream=False, mode=Mode.TOOLS
        )


@pytest.mark.asyncio
async def test_unsupported_provider_surfaces_registry_error_sync_and_async() -> None:
    raw = chat_completion(content='{"name": "Ada", "age": 36}')

    with pytest.raises(KeyError, match="not registered"):
        process_response(
            raw,
            response_model=User,
            stream=False,
            mode=Mode.JSON,
            provider=Provider.UNKNOWN,
        )

    with pytest.raises(KeyError, match="not registered"):
        await process_response_async(
            raw,
            response_model=User,
            stream=False,
            mode=Mode.JSON,
            provider=Provider.UNKNOWN,
        )


@pytest.mark.asyncio
async def test_non_choice_iterable_response_becomes_list_response_sync_and_async() -> (
    None
):
    raw = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text='{"tasks": [{"name": "Ada", "age": 36}]}')
        ]
    )

    sync_result = process_response(
        cast(Any, raw),
        response_model=UsersEnvelope,
        stream=False,
        strict=True,
        mode=Mode.JSON,
    )
    async_result = await process_response_async(
        cast(Any, raw),
        response_model=UsersEnvelope,
        stream=False,
        strict=True,
        mode=Mode.JSON,
    )

    assert isinstance(sync_result, ListResponse)
    assert isinstance(async_result, ListResponse)
    assert list(sync_result) == [User(name="Ada", age=36)]
    assert list(async_result) == [User(name="Ada", age=36)]
    assert sync_result.get_raw_response() is raw
    assert async_result.get_raw_response() is raw


@pytest.mark.asyncio
async def test_iterable_completion_list_is_wrapped_sync_and_async() -> None:
    response_model = IterableModel(User)
    raw = chat_completion(
        tool_calls=[
            tool_call(
                response_model.__name__,
                {"tasks": [{"name": "Ada", "age": 36}, {"name": "Lin", "age": 29}]},
            )
        ]
    )

    sync_result = process_response(
        raw, response_model=response_model, stream=False, mode=Mode.TOOLS
    )
    async_result = await process_response_async(
        raw, response_model=response_model, stream=False, mode=Mode.TOOLS
    )

    expected = [User(name="Ada", age=36), User(name="Lin", age=29)]
    assert isinstance(sync_result, ListResponse)
    assert isinstance(async_result, ListResponse)
    assert list(sync_result) == expected
    assert list(async_result) == expected
    assert sync_result.get_raw_response() is raw
    assert async_result.get_raw_response() is raw


@pytest.mark.asyncio
async def test_partial_sync_stream_is_preserved_by_sync_and_async_conversion() -> None:
    response_model = Partial[User]
    chunks = tool_chunks('{"name": "Ad', 'a", "age": 36}')

    sync_result = process_response(
        cast(Any, chunks), response_model=response_model, stream=True, mode=Mode.TOOLS
    )
    async_result = await process_response_async(
        cast(Any, chunks), response_model=response_model, stream=True, mode=Mode.TOOLS
    )

    expected = [User.model_construct(name="Ad", age=None), User(name="Ada", age=36)]
    assert list(sync_result) == expected
    assert list(async_result) == expected


def test_partial_sync_accepts_a_materialized_provider_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = chat_completion(content="{}")
    snapshots = [User(name="Ada", age=36)]

    def parser(**_kwargs: Any) -> list[User]:
        return snapshots

    _install_parser(monkeypatch, parser)
    assert (
        process_response(
            raw,
            response_model=Partial[User],
            stream=True,
            mode=Mode.JSON,
            provider=Provider.UNKNOWN,
        )
        is snapshots
    )


@pytest.mark.asyncio
async def test_async_iterable_stream_remains_an_async_generator() -> None:
    response_model = IterableModel(User)
    raw = async_items(
        tool_chunks(
            '{"tasks": [{"name": "Ada", "age": 36},',
            ' {"name": "Lin", "age": 29}]}',
        )
    )

    result = await process_response_async(
        cast(Any, raw), response_model=response_model, stream=True, mode=Mode.TOOLS
    )
    users = [user async for user in result]

    assert users == [User(name="Ada", age=36), User(name="Lin", age=29)]


@pytest.mark.asyncio
async def test_custom_iterable_parser_is_wrapped_sync_and_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = chat_completion(content='{"tasks": [{"name": "Ada", "age": 36}]}')

    def parser(*, response: ChatCompletion, **_kwargs: Any) -> UsersEnvelope:
        content = response.choices[0].message.content
        assert content is not None
        return UsersEnvelope.model_validate_json(content)

    _install_parser(monkeypatch, parser)

    sync_result = process_response(
        raw,
        response_model=UsersEnvelope,
        stream=False,
        mode=Mode.JSON,
        provider=Provider.UNKNOWN,
    )
    async_result = await process_response_async(
        raw,
        response_model=UsersEnvelope,
        stream=False,
        mode=Mode.JSON,
        provider=Provider.UNKNOWN,
    )

    assert isinstance(sync_result, ListResponse)
    assert isinstance(async_result, ListResponse)
    assert list(sync_result) == [User(name="Ada", age=36)]
    assert list(async_result) == [User(name="Ada", age=36)]
    assert sync_result.get_raw_response() is raw
    assert async_result.get_raw_response() is raw


@pytest.mark.asyncio
async def test_custom_parallel_parser_preserves_results_and_raw_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = chat_completion(
        tool_calls=[
            tool_call("User", {"name": "Ada", "age": 36}, "call-ada"),
            tool_call("User", {"name": "Lin", "age": 29}, "call-lin"),
        ]
    )
    response_model = ParallelBase(User)

    def parser(
        *, response: ChatCompletion, response_model: ParallelBase[User], **_kwargs: Any
    ) -> ListResponse[User]:
        models = list(response_model.from_response(response, mode=Mode.JSON))
        return ListResponse.from_list(models, raw_response=None)

    _install_parser(monkeypatch, parser)

    sync_result = process_response(
        raw,
        response_model=cast(Any, response_model),
        stream=False,
        mode=Mode.JSON,
        provider=Provider.UNKNOWN,
    )
    async_result = await process_response_async(
        raw,
        response_model=cast(Any, response_model),
        stream=False,
        mode=Mode.JSON,
        provider=Provider.UNKNOWN,
    )

    expected = [User(name="Ada", age=36), User(name="Lin", age=29)]
    assert list(sync_result) == expected
    assert list(async_result) == expected
    assert sync_result.get_raw_response() is raw
    assert async_result.get_raw_response() is raw


@pytest.mark.asyncio
async def test_custom_adapter_parser_returns_simple_value_sync_and_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = chat_completion(content='{"content": 42}')

    def parser(*, response: ChatCompletion, **_kwargs: Any) -> IntegerAdapter:
        content = response.choices[0].message.content
        assert content is not None
        return IntegerAdapter.model_validate_json(content)

    _install_parser(monkeypatch, parser)

    assert (
        process_response(
            raw,
            response_model=IntegerAdapter,
            stream=False,
            mode=Mode.JSON,
            provider=Provider.UNKNOWN,
        )
        == 42
    )
    assert (
        await process_response_async(
            raw,
            response_model=IntegerAdapter,
            stream=False,
            mode=Mode.JSON,
            provider=Provider.UNKNOWN,
        )
        == 42
    )


@pytest.mark.asyncio
async def test_custom_parser_can_return_native_mapping_sync_and_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = chat_completion(content='{"name": "Ada", "age": 36}')

    def parser(*, response: ChatCompletion, **_kwargs: Any) -> dict[str, Any]:
        content = response.choices[0].message.content
        assert content is not None
        return json.loads(content)

    _install_parser(monkeypatch, parser)

    expected = {"name": "Ada", "age": 36}
    assert (
        process_response(
            raw,
            response_model=User,
            stream=False,
            mode=Mode.JSON,
            provider=Provider.UNKNOWN,
        )
        == expected
    )
    assert (
        await process_response_async(
            raw,
            response_model=User,
            stream=False,
            mode=Mode.JSON,
            provider=Provider.UNKNOWN,
        )
        == expected
    )


def test_handle_response_model_without_schema_preserves_request_kwargs() -> None:
    messages = [{"role": "user", "content": "Hello"}]

    response_model, kwargs = handle_response_model(
        None,
        mode=Mode.TOOLS,
        provider=Provider.OPENAI,
        messages=messages,
    )

    assert response_model is None
    assert kwargs == {"messages": messages}
    assert kwargs["messages"] is not messages


def test_is_typed_dict_distinguishes_typed_dicts_from_other_inputs() -> None:
    class UserDict(TypedDict):
        name: str
        age: int

    assert is_typed_dict(UserDict)
    assert not is_typed_dict(User)
    assert not is_typed_dict({"name": "Ada", "age": 36})
