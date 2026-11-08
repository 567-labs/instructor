"""JSON/MD_JSON schema instruction must target a text part of the system message.

The JSON and MD_JSON request handlers append the generated schema instruction
to the leading system message. When that message's content is a content-part
list, the instruction must merge into the first *text* part; it must not
assume the first part is text (an image or audio part may come first) nor
crash when no text part exists.
"""

from __future__ import annotations

from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry


class Answer(BaseModel):
    answer: float


IMAGE_PART = {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}


def _prepared_system_content(provider: Provider, mode: Mode, content) -> list:
    kwargs = {
        "messages": [
            {"role": "system", "content": content},
            {"role": "user", "content": "What is 2+2?"},
        ]
    }
    _, result_kwargs = mode_registry.get_handlers(provider, mode).request_handler(
        Answer, kwargs
    )
    return result_kwargs["messages"][0]["content"]


def test_openai_json_merges_into_text_part_after_image() -> None:
    content = _prepared_system_content(
        Provider.OPENAI,
        Mode.JSON,
        [IMAGE_PART, {"type": "text", "text": "You are helpful."}],
    )

    assert content[0] == IMAGE_PART
    assert "You are helpful." in content[1]["text"]
    assert "json_schema" in content[1]["text"]


def test_openai_md_json_merges_into_text_part_after_image() -> None:
    content = _prepared_system_content(
        Provider.OPENAI,
        Mode.MD_JSON,
        [IMAGE_PART, {"type": "text", "text": "You are helpful."}],
    )

    assert content[0] == IMAGE_PART
    assert "You are helpful." in content[1]["text"]
    assert "json_schema" in content[1]["text"]


def test_openai_json_adds_text_part_when_missing() -> None:
    content = _prepared_system_content(Provider.OPENAI, Mode.JSON, [IMAGE_PART])

    assert content[0] == IMAGE_PART
    assert len(content) == 2
    assert "json_schema" in content[1]["text"]


def test_openai_json_adds_text_part_to_empty_list() -> None:
    content = _prepared_system_content(Provider.OPENAI, Mode.JSON, [])

    assert len(content) == 1
    assert "json_schema" in content[0]["text"]


def test_openai_json_keeps_string_system_message() -> None:
    content = _prepared_system_content(Provider.OPENAI, Mode.JSON, "You are helpful.")

    assert isinstance(content, str)
    assert "You are helpful." in content
    assert "json_schema" in content


def test_mistral_md_json_merges_into_text_part_after_image() -> None:
    content = _prepared_system_content(
        Provider.MISTRAL,
        Mode.MD_JSON,
        [IMAGE_PART, {"type": "text", "text": "You are helpful."}],
    )

    assert content[0] == IMAGE_PART
    assert "json_schema" in content[1]["text"]


def test_writer_md_json_merges_into_text_part_after_image() -> None:
    content = _prepared_system_content(
        Provider.WRITER,
        Mode.MD_JSON,
        [IMAGE_PART, {"type": "text", "text": "You are helpful."}],
    )

    assert content[0] == IMAGE_PART
    assert "json_schema" in content[1]["text"]


def test_xai_md_json_merges_into_text_part_after_image() -> None:
    content = _prepared_system_content(
        Provider.XAI,
        Mode.MD_JSON,
        [IMAGE_PART, {"type": "text", "text": "You are helpful."}],
    )

    assert content[0] == IMAGE_PART
    assert "json_schema" in content[1]["text"]
