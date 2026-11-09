from __future__ import annotations

from typing import Any

import pytest
from jinja2 import UndefinedError

from instructor import Mode
from instructor.v2.core.templating import handle_templating


def test_unmatched_template_variable_raises():
    """An unmatched {{ }} in the message should raise, not silently disappear."""
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "Explain what {{ jinja_example }} means, and greet {{ user_name }}.",
            },
        ],
    }

    with pytest.raises(UndefinedError):
        handle_templating(kwargs, Mode.TOOLS, context={"user_name": "Ada"})


def test_raw_block_round_trips_literal_braces():
    """{% raw %}...{% endraw %} should still let literal {{ }} text pass through untouched."""
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "Here is a literal example: {% raw %}{{ jinja_example }}{% endraw %}, greet {{ user_name }}.",
            },
        ],
    }

    result = handle_templating(kwargs, Mode.TOOLS, context={"user_name": "Ada"})

    assert (
        result["messages"][0]["content"]
        == "Here is a literal example: {{ jinja_example }}, greet Ada."
    )


@pytest.mark.parametrize("context", [{}, {"unused": True}])
def test_explicit_context_renders_templates_without_variables(
    context: dict[str, Any],
) -> None:
    kwargs = {"messages": [{"role": "user", "content": "Return {{ 2 + 2 }}."}]}
    result = handle_templating(kwargs, Mode.TOOLS, context=context)
    assert result["messages"][0]["content"] == "Return 4."
    assert kwargs["messages"][0]["content"] == "Return {{ 2 + 2 }}."


def test_no_context_preserves_literal_template_text() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Return {{ missing }}."}]}
    assert handle_templating(kwargs, Mode.TOOLS, context=None) is kwargs


def test_legacy_empty_context_renders_raw_blocks() -> None:
    kwargs = {
        "messages": [{"role": "user", "content": "{% raw %}{{ literal }}{% endraw %}"}]
    }
    result = handle_templating(kwargs, Mode.TOOLS, {})
    assert result["messages"][0]["content"] == "{{ literal }}"


def test_sync_client_rejects_missing_variables_with_empty_context() -> None:
    from instructor import from_openai
    from openai import OpenAI
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    # Missing variables must fail before the SDK can contact this closed local port.
    with OpenAI(
        api_key="test-not-a-key",
        base_url="http://127.0.0.1:9/v1",
        max_retries=0,
        timeout=0.1,
    ) as sdk:
        client = from_openai(sdk, mode=Mode.JSON)
        with pytest.raises(UndefinedError, match="missing"):
            client.create(
                model="test",
                response_model=Answer,
                messages=[{"role": "user", "content": "Return {{ missing }}."}],
                context={},
                max_retries=1,
            )


@pytest.mark.asyncio
async def test_async_client_rejects_missing_variables_with_empty_context() -> None:
    from instructor import from_openai
    from openai import AsyncOpenAI
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    async with AsyncOpenAI(
        api_key="test-not-a-key",
        base_url="http://127.0.0.1:9/v1",
        max_retries=0,
        timeout=0.1,
    ) as sdk:
        client = from_openai(sdk, mode=Mode.JSON)
        with pytest.raises(UndefinedError, match="missing"):
            await client.create(
                model="test",
                response_model=Answer,
                messages=[{"role": "user", "content": "Return {{ missing }}."}],
                context={},
                max_retries=1,
            )
