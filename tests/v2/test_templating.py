from __future__ import annotations

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


def test_explicit_empty_context_still_renders():
    """An explicit empty context renders; only None means "no templating"."""
    kwargs = {"messages": [{"role": "user", "content": "{{ 6 * 7 }}"}]}

    rendered = handle_templating(kwargs, Mode.TOOLS, context={})
    untemplated = handle_templating(kwargs, Mode.TOOLS, context=None)

    assert rendered["messages"][0]["content"] == "42"
    assert untemplated["messages"][0]["content"] == "{{ 6 * 7 }}"
