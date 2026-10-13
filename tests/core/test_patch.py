import functools
import inspect
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
import pytest

import instructor
from instructor.utils import is_async


def test_patch_completes_successfully():
    with OpenAI(api_key="test-key") as client:
        instructor.patch(client)


@pytest.mark.asyncio
async def test_apatch_completes_successfully():
    async with AsyncOpenAI(api_key="test-key") as client:
        with pytest.warns(
            DeprecationWarning, match="apatch is deprecated, use patch instead"
        ):
            instructor.apatch(client)


def test_is_async_returns_true_if_function_is_async():
    async def async_function():
        pass

    assert is_async(async_function) is True


def test_is_async_returns_false_if_function_is_not_async():
    def sync_function():
        pass

    assert is_async(sync_function) is False


def test_is_async_returns_true_if_wrapped_function_is_async():
    async def async_function():
        pass

    @functools.wraps(async_function)
    def wrapped_function():
        pass

    assert is_async(wrapped_function) is True


def test_is_async_returns_true_if_double_wrapped_function_is_async():
    async def async_function():
        pass

    @functools.wraps(async_function)
    def wrapped_function():
        pass

    @functools.wraps(wrapped_function)
    def double_wrapped_function():
        pass

    assert is_async(double_wrapped_function) is True


def test_is_async_returns_true_if_triple_wrapped_function_is_async():
    async def async_function():
        pass

    @functools.wraps(async_function)
    def wrapped_function():
        pass

    @functools.wraps(wrapped_function)
    def double_wrapped_function():
        pass

    @functools.wraps(double_wrapped_function)
    def triple_wrapped_function():
        pass

    assert is_async(triple_wrapped_function) is True


def test_is_async_returns_false_for_sync_callable_instance():
    class SyncCreate:
        def __call__(self) -> None:
            pass

    assert is_async(SyncCreate()) is False


def test_is_async_returns_false_for_constructor_of_async_callable():
    class AsyncCreate:
        async def __call__(self) -> None:
            pass

    assert is_async(AsyncCreate) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_patch_async_callable_instance_parses_response(wrapped: bool):
    class Person(BaseModel):
        name: str

    class AsyncCreate:
        async def __call__(self, **_kwargs: Any) -> ChatCompletion:
            return ChatCompletion.model_validate(
                {
                    "id": "test-completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"name": "Ada"}',
                            },
                        }
                    ],
                }
            )

    create = AsyncCreate()
    if wrapped:

        @functools.wraps(create)
        def wrapped_create(**kwargs: Any):
            return create(**kwargs)

        patched = instructor.patch(create=wrapped_create, mode=instructor.Mode.JSON)
    else:
        patched = instructor.patch(create=create, mode=instructor.Mode.JSON)

    assert inspect.iscoroutinefunction(patched)
    person = await patched(
        response_model=Person,
        model="test-model",
        messages=[{"role": "user", "content": "Extract Ada's name."}],
    )
    assert person.name == "Ada"
