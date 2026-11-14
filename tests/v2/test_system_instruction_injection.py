"""JSON / MD_JSON schema injection into an existing system message (#3049)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry


class Answer(BaseModel):
    favorite_color_name: str


IMAGE_PART = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
TEXT_PART = {"type": "text", "text": "You are helpful."}

PAIRS = [
    (Provider.OPENAI, Mode.JSON),
    (Provider.OPENAI, Mode.MD_JSON),
    (Provider.WRITER, Mode.MD_JSON),
    (Provider.MISTRAL, Mode.MD_JSON),
    (Provider.XAI, Mode.MD_JSON),
]
CONTENTS: dict[str, Any] = {
    "string": "You are helpful.",
    "none": None,
    "empty_list": [],
    "image_then_text": [IMAGE_PART, TEXT_PART],
    "image_only": [IMAGE_PART],
}


@pytest.mark.parametrize(
    ("provider", "mode"),
    PAIRS,
    ids=[f"{p.value}-{m.value}" for p, m in PAIRS],
)
@pytest.mark.parametrize("content_name", list(CONTENTS))
def test_schema_injected_into_existing_system_message(
    provider: Provider, mode: Mode, content_name: str
) -> None:
    original = CONTENTS[content_name]
    caller_messages = [
        {"role": "system", "content": deepcopy(original)},
        {"role": "user", "content": "hi"},
    ]
    snapshot = deepcopy(caller_messages)

    handlers = mode_registry.get_handlers(provider, mode)
    _, new_kwargs = handlers.request_handler(
        response_model=Answer,
        kwargs={"model": "test", "messages": caller_messages},
    )

    system_messages = [m for m in new_kwargs["messages"] if m["role"] == "system"]
    assert len(system_messages) == 1
    content = system_messages[0]["content"]
    if isinstance(original, list):
        assert isinstance(content, list)
        text = "".join(p["text"] for p in content if p.get("type") == "text")
        if IMAGE_PART in original:
            assert content[0] == IMAGE_PART
    else:
        assert isinstance(content, str)
        text = content
    assert "favorite_color_name" in text
    if original and original != [IMAGE_PART]:
        assert "You are helpful." in text
    assert caller_messages == snapshot
