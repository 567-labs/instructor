from copy import deepcopy

import pytest
from jinja2 import UndefinedError

from instructor import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.templating import handle_templating


@pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.AZURE_OPENAI])
@pytest.mark.parametrize("messages", [[], [{"role": "user", "content": "{{ name }}"}]])
def test_response_instructions_render_without_mutating_input(provider, messages):
    kwargs = {"instructions": "Extract {{ field }}", "messages": messages}
    before = deepcopy(kwargs)
    result = handle_templating(
        kwargs,
        Mode.RESPONSES_TOOLS,
        provider=provider,
        context={"field": "name", "name": "Ada"},
    )
    assert result["instructions"] == "Extract name"
    assert kwargs == before
    if messages:
        assert result["messages"][0]["content"] == "Ada"


def test_response_instructions_missing_variable_raises():
    with pytest.raises(UndefinedError):
        handle_templating(
            {"instructions": "{{ missing }}"},
            Mode.RESPONSES_TOOLS,
            context={"name": "Ada"},
        )


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
def test_other_modes_preserve_instructions(mode):
    result = handle_templating(
        {"instructions": "{{ missing }}"}, mode, context={"name": "Ada"}
    )
    assert result["instructions"] == "{{ missing }}"


@pytest.mark.parametrize("value", [None, 1, ["{{ name }}"]])
def test_non_text_instructions_remain_unchanged(value):
    result = handle_templating(
        {"instructions": value}, Mode.RESPONSES_TOOLS, context={"name": "Ada"}
    )
    assert result["instructions"] == value


@pytest.mark.parametrize("context", [None, {}])
def test_without_context_preserves_literal_instructions(context):
    kwargs = {"instructions": "{{ missing }}"}
    assert handle_templating(kwargs, Mode.RESPONSES_TOOLS, context=context) is kwargs


def test_response_instructions_preserve_explicit_raw_block():
    result = handle_templating(
        {"instructions": "{% raw %}{{ literal }}{% endraw %} {{ name }}"},
        Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        context={"name": "Ada"},
    )
    assert result["instructions"] == "{{ literal }} Ada"
