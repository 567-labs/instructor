import functools

from openai import AsyncOpenAI, OpenAI
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


def test_is_async_returns_true_for_async_callable_object():
    class AsyncCreate:
        async def __call__(self, **kwargs):
            pass

    assert is_async(AsyncCreate()) is True


def test_is_async_returns_true_for_wrapped_async_callable_object():
    class AsyncCreate:
        async def __call__(self, **kwargs):
            pass

    create = AsyncCreate()

    @functools.wraps(create)
    def wrapped(**kwargs):
        return create(**kwargs)

    assert is_async(wrapped) is True


def test_is_async_returns_false_for_sync_callable_object():
    class SyncCreate:
        def __call__(self, **kwargs):
            pass

    assert is_async(SyncCreate()) is False


def test_is_async_returns_false_for_class_with_async_call():
    class AsyncCreate:
        async def __call__(self, **kwargs):
            pass

    # Inspecting the class itself should not report True since calling the class invokes type.__call__
    assert is_async(AsyncCreate) is False


def test_patch_async_callable_object_creates_async_wrapper():
    import inspect

    class AsyncCreate:
        async def __call__(self, **kwargs):
            pass

    create = AsyncCreate()
    patched = instructor.patch(create=create, mode=instructor.Mode.JSON)
    assert inspect.iscoroutinefunction(patched) is True


@pytest.mark.asyncio
async def test_patch_async_callable_object_executes_and_parses():
    from unittest.mock import MagicMock
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = '{"name": "Alice", "age": 30}'
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None

    class MockAsyncCreate:
        async def __call__(self, *args, **kwargs):
            del args, kwargs
            return mock_completion

    create = MockAsyncCreate()
    patched = instructor.patch(create=create, mode=instructor.Mode.JSON)

    result = await patched(
        response_model=User,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Extract"}],
    )
    assert isinstance(result, User)
    assert result.name == "Alice"
    assert result.age == 30
