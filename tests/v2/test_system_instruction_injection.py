"""JSON / MD_JSON schema injection into list-content system messages (#3049)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.messages import append_system_instruction
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
CONTENTS: dict[str, list[dict[str, Any]]] = {
    "empty": [],
    "image_then_text": [IMAGE_PART, TEXT_PART],
    "image_only": [IMAGE_PART],
}


@pytest.mark.parametrize(
    ("provider", "mode"),
    PAIRS,
    ids=[f"{p.value}-{m.value}" for p, m in PAIRS],
)
@pytest.mark.parametrize("content_name", list(CONTENTS))
def test_schema_injected_into_list_system_content(
    provider: Provider, mode: Mode, content_name: str
) -> None:
    caller_messages = [
        {"role": "system", "content": deepcopy(CONTENTS[content_name])},
        {"role": "user", "content": "hi"},
    ]
    snapshot = deepcopy(caller_messages)

    handlers = mode_registry.get_handlers(provider, mode)
    _, new_kwargs = handlers.request_handler(
        response_model=Answer,
        kwargs={"model": "test", "messages": caller_messages},
    )

    messages = new_kwargs["messages"]
    system_messages = [m for m in messages if m["role"] == "system"]
    assert len(system_messages) == 1
    content = system_messages[0]["content"]
    assert isinstance(content, list)
    text_parts = [p for p in content if p.get("type") == "text"]
    assert any("favorite_color_name" in p["text"] for p in text_parts)
    if IMAGE_PART in CONTENTS[content_name]:
        assert content[0] == IMAGE_PART
    assert caller_messages == snapshot


def test_append_system_instruction_shapes() -> None:
    message: dict[str, Any] = {"role": "system", "content": "base"}
    append_system_instruction(message, "extra")
    assert message["content"] == "base\n\nextra"

    message = {"role": "system", "content": [IMAGE_PART, dict(TEXT_PART)]}
    append_system_instruction(message, "extra")
    assert message["content"] == [
        IMAGE_PART,
        {"type": "text", "text": "You are helpful.\n\nextra"},
    ]

    message = {"role": "system", "content": [IMAGE_PART]}
    append_system_instruction(message, "extra")
    assert message["content"] == [IMAGE_PART, {"type": "text", "text": "extra"}]


def test_append_system_instruction_replaces_missing_content() -> None:
    message: dict[str, Any] = {"role": "system", "content": None}
    append_system_instruction(message, "extra")
    assert message["content"] == "extra"
