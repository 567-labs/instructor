"""
Response mode tests that run across all core providers.

Tests different response modes and methods available on the client.
"""

from pydantic import BaseModel
import pytest
import instructor

from .capabilities import skip_if_unsupported


class Task(BaseModel):
    title: str
    description: str
    priority: int


@pytest.mark.asyncio
async def test_create_method(provider_config):
    """Test the create() method."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    task = await client.create(
        response_model=Task,
        messages=[
            {
                "role": "user",
                "content": "Create a task: Fix bug in login, high priority (9)",
            }
        ],
    )

    assert isinstance(task, Task)
    assert "bug" in task.title.lower() or "login" in task.title.lower()
    assert task.priority == 9


@pytest.mark.asyncio
async def test_chat_completions_create_method(provider_config):
    """Test the chat.completions.create() method."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    task = await client.chat.completions.create(
        response_model=Task,
        messages=[
            {
                "role": "user",
                "content": "Task: Update documentation, medium priority (5)",
            }
        ],
    )

    assert isinstance(task, Task)
    assert task.priority == 5


@pytest.mark.asyncio
async def test_messages_create_method(provider_config):
    """Test the messages.create() method."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    task = await client.messages.create(
        response_model=Task,
        messages=[
            {
                "role": "user",
                "content": "Task: Review PR, low priority (3)",
            }
        ],
    )

    assert isinstance(task, Task)
    assert task.priority == 3


@pytest.mark.asyncio
async def test_create_with_completion(provider_config):
    """Test create_with_completion() returns both model and raw response."""
    skip_if_unsupported(provider_config, "create_with_completion")
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    task, completion = await client.chat.completions.create_with_completion(
        response_model=Task,
        messages=[
            {
                "role": "user",
                "content": "Task: Deploy to production, priority 10",
            }
        ],
    )

    assert isinstance(task, Task)
    assert task.priority == 10
    # completion should be the raw response object from the provider
    assert completion is not None


@pytest.mark.asyncio
async def test_response_model_none(provider_config):
    """Test that response_model=None returns raw response."""
    skip_if_unsupported(provider_config, "response_model_none")
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    response = await client.messages.create(
        response_model=None,
        messages=[{"role": "user", "content": "Say hello!"}],
    )

    # Should return raw provider response
    assert response is not None
