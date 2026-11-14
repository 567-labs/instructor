"""Anthropic/Bedrock top-level ``system`` is templated as a string or block list."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from instructor import Mode, Provider
from instructor.v2.core.templating import handle_templating


@pytest.mark.parametrize("provider", [Provider.ANTHROPIC, Provider.BEDROCK])
def test_string_system_is_rendered(provider: Provider) -> None:
    kwargs: dict[str, Any] = {
        "system": "You help {{ name }}.",
        "messages": [{"role": "user", "content": "Hi {{ name }}"}],
    }
    snapshot = copy.deepcopy(kwargs)

    result = handle_templating(kwargs, Mode.TOOLS, provider, {"name": "Ada"})

    assert result["system"] == "You help Ada."
    assert kwargs == snapshot


def test_list_system_is_rendered() -> None:
    kwargs: dict[str, Any] = {
        "system": [{"type": "text", "text": "You help {{ name }}."}],
        "messages": [{"role": "user", "content": "Hi"}],
    }
    snapshot = copy.deepcopy(kwargs)

    result = handle_templating(kwargs, Mode.TOOLS, Provider.ANTHROPIC, {"name": "Ada"})

    assert result["system"] == [{"type": "text", "text": "You help Ada."}]
    assert kwargs == snapshot
