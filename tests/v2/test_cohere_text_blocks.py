"""Parse complete Cohere V2 text without discarding later content blocks."""

from __future__ import annotations

from typing import Any

import pytest

from instructor import Mode, OpenAISchema, Provider
from instructor.core.exceptions import ResponseParsingError

cohere = pytest.importorskip("cohere", minversion="5.17.0")


class Answer(OpenAISchema):
    answer: str


MODES = [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON]


def response_with_blocks(blocks: list[Any] | None) -> cohere.ChatResponse:
    return cohere.ChatResponse(
        id="cohere-text-blocks",
        finish_reason="COMPLETE",
        message=cohere.AssistantMessageResponse(content=blocks),
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "fragments",
    [
        ['{"answer":', '"café"}'],
        ["", '{"answer":"café"}'],
        ['{"ans', 'wer":"caf', 'é"}'],
    ],
)
def test_parse_all_text_blocks(mode: Mode, fragments: list[str]) -> None:
    response = response_with_blocks(
        [
            cohere.TextAssistantMessageResponseContentItem(text=text)
            for text in fragments
        ]
    )
    original = response.model_dump()

    result = Answer.from_response(response, mode=mode, provider=Provider.COHERE)

    assert result.answer == "café"
    assert response.model_dump() == original


@pytest.mark.parametrize("mode", MODES)
def test_skip_thinking_between_text_blocks(mode: Mode) -> None:
    response = response_with_blocks(
        [
            cohere.ThinkingAssistantMessageResponseContentItem(value="not JSON"),
            cohere.TextAssistantMessageResponseContentItem(text='{"answer":'),
            cohere.ThinkingAssistantMessageResponseContentItem(value="still not JSON"),
            cohere.TextAssistantMessageResponseContentItem(text='"café"}'),
        ]
    )

    result = Answer.from_response(response, mode=mode, provider=Provider.COHERE)

    assert result.answer == "café"


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.MD_JSON])
def test_markdown_fence_spans_text_blocks(mode: Mode) -> None:
    response = response_with_blocks(
        [
            cohere.TextAssistantMessageResponseContentItem(text="```json\n"),
            cohere.TextAssistantMessageResponseContentItem(text='{"answer":"café"}'),
            cohere.TextAssistantMessageResponseContentItem(text="\n```"),
        ]
    )

    result = Answer.from_response(response, mode=mode, provider=Provider.COHERE)

    assert result.answer == "café"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_single_text_response_remains_supported(mode: Mode, version: str) -> None:
    text = '{"answer":"café"}'
    response = (
        cohere.NonStreamedChatResponse(text=text, finish_reason="COMPLETE")
        if version == "v1"
        else response_with_blocks(
            [cohere.TextAssistantMessageResponseContentItem(text=text)]
        )
    )

    result = Answer.from_response(response, mode=mode, provider=Provider.COHERE)

    assert result.answer == "café"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "blocks",
    [None, [], [cohere.ThinkingAssistantMessageResponseContentItem(value="reasoning")]],
)
def test_response_without_text_remains_an_error(
    mode: Mode, blocks: list[Any] | None
) -> None:
    response = response_with_blocks(blocks)

    with pytest.raises(ResponseParsingError) as exc_info:
        Answer.from_response(response, mode=mode, provider=Provider.COHERE)

    assert exc_info.value.raw_response is response
