from __future__ import annotations

import pytest
from jinja2 import UndefinedError

from instructor import Mode
from instructor.v2.core.templating import apply_template, handle_templating


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("  Static text {value}", "Static text {value}"),
        ("  first\r\n  second\r\n", "first\nsecond"),
        ("  first\r  second\r\r", "first\nsecond\n"),
        ("a\u2028b\n", "a\u2028b"),
        ("A{# hidden #}B", "AB"),
    ],
)
def test_apply_template_preserves_literal_text_formatting(
    text: str, expected: str
) -> None:
    assert apply_template(text, {"unused": "value"}) == expected


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
