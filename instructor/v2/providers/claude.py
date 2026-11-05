"""Claude tool-choice rules shared by Anthropic and Bedrock Converse."""

from __future__ import annotations

import re


def requires_auto_tool_choice(model: object, thinking: object) -> bool:
    if isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive"):
        return True
    return (
        isinstance(model, str)
        and re.search(
            r"(?:^|[./:])claude-(?:(?:opus|sonnet)-5[.-]5|(?:fable|mythos)-5[.-]1)(?:$|[-:@])",
            model,
            flags=re.IGNORECASE,
        )
        is not None
    )
