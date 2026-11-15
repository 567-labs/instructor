"""Contract: structured output is read from text content blocks only.

Text blocks are joined, reasoning/thinking content is skipped, and a reply
without text is an error. Covers Cohere/OpenAI-compatible/Mistral content
blocks, Mistral ThinkChunk streams, GenAI thought parts and empty chunks, and
OpenAI Responses ``commentary``-phase messages. All inputs are real SDK
objects fed straight into the registered parser.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Optional

import pytest
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel, ValidationError

import instructor
from instructor.v2.core.errors import ResponseParsingError

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.partial import Partial
from tests import openai_http as httpx
from tests._streams import async_items
from tests.coverage._openai import chat_completion


class Item(BaseModel):
    name: str


async def _stream(
    provider: Provider, mode: Mode, model: Any, chunks: list[Any], is_async: bool
) -> list[Any]:
    parser: Any = mode_registry.get_handlers(provider, mode).response_parser
    if is_async:
        stream = parser(async_items(chunks), model, stream=True, is_async=True)
        return [item async for item in stream]
    return list(parser(iter(chunks), model, stream=True))


sync_and_async = pytest.mark.parametrize(
    "is_async", [False, True], ids=["sync", "async"]
)

# --- Mistral: content as [ThinkChunk, TextChunk] ---------------------------


@sync_and_async
@pytest.mark.asyncio
async def test_mistral_stream_list_deltas_skip_thinking(is_async: bool) -> None:
    m = pytest.importorskip("mistralai.client.models")

    def event(content: Any) -> Any:
        choice = m.CompletionResponseStreamChoice(
            index=0, delta=m.DeltaMessage(content=content), finish_reason=None
        )
        return m.CompletionEvent(
            data=m.CompletionChunk(id="c", model="magistral", choices=[choice])
        )

    chunks = [
        event([m.ThinkChunk(thinking=[m.TextChunk(text="thinking")])]),
        event([m.TextChunk(text='{"tasks": [{"name": ')]),
        event(None),
        event('"Ada"}]}'),
    ]

    received = await _stream(
        Provider.MISTRAL, Mode.JSON_SCHEMA, IterableModel(Item), chunks, is_async
    )
    assert received == [Item(name="Ada")]


# --- GenAI: thought parts and chunks without parts -------------------------


def _genai_chunk(parts: list[Any] | None) -> Any:
    types = pytest.importorskip("google.genai.types")
    content = types.Content(role="model", parts=parts)
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def _genai_call(names: list[str]) -> Any:
    types = pytest.importorskip("google.genai.types")
    args = {"tasks": [{"name": n} for n in names]}
    return types.Part(function_call=types.FunctionCall(name="IterableItem", args=args))


def _genai_thought() -> Any:
    types = pytest.importorskip("google.genai.types")
    return types.Part(thought=True, text='checking {"name": "x"}')


def _genai_text(text: str) -> Any:
    types = pytest.importorskip("google.genai.types")
    return types.Part(text=text)


@sync_and_async
@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.asyncio
async def test_genai_stream_skips_thought_parts_and_empty_chunks(
    mode: Mode, is_async: bool
) -> None:
    empty = [_genai_chunk(None), _genai_chunk([]), _genai_chunk([_genai_thought()])]
    if mode is Mode.TOOLS:
        chunks = [*empty, _genai_chunk([_genai_call(["Ada", "Bob"])])]
    else:
        chunks = [
            empty[0],
            _genai_chunk([_genai_text('{"tasks": [{"name": "Ada"}, ')]),
            empty[1],
            empty[2],
            _genai_chunk([_genai_text('{"name": "Bob"}]}')]),
        ]

    received = await _stream(
        Provider.GENAI, mode, IterableModel(Item), chunks, is_async
    )
    assert received == [Item(name="Ada"), Item(name="Bob")]


@sync_and_async
@pytest.mark.parametrize("thought_first", [True, False], ids=["before", "after"])
@pytest.mark.asyncio
async def test_genai_tools_stream_uses_call_next_to_thought_part(
    thought_first: bool, is_async: bool
) -> None:
    parts = [_genai_thought(), _genai_call(["Ada"])]
    chunk = _genai_chunk(parts if thought_first else parts[::-1])

    received = await _stream(
        Provider.GENAI, Mode.TOOLS, IterableModel(Item), [chunk], is_async
    )
    assert received == [Item(name="Ada")]


# --- OpenAI Responses: commentary-phase message precedes the answer --------


def test_openai_responses_stream_skips_commentary_message() -> None:
    from openai.types.responses import (
        ResponseOutputItemAddedEvent,
        ResponseOutputMessage,
        ResponseTextDeltaEvent,
    )

    def message(index: int, phase: Any, text: str) -> list[Any]:
        item = ResponseOutputMessage(
            id=f"msg-{index}",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
            phase=phase,
        )
        return [
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                output_index=index,
                sequence_number=index * 2,
                item=item,
            ),
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                item_id=f"msg-{index}",
                output_index=index,
                content_index=0,
                sequence_number=index * 2 + 1,
                delta=text,
                logprobs=[],
            ),
        ]

    events = [
        *message(0, "commentary", 'Checking {"name": "x"} first.'),
        *message(1, "final_answer", '{"name": "Ada"}'),
    ]
    handlers = mode_registry.get_handlers(Provider.OPENAI, Mode.RESPONSES_TOOLS)
    parser: Any = handlers.response_parser
    results = list(parser(iter(events), Partial[Item], stream=True))

    assert results[-1].name == "Ada"


# --- Non-streaming text blocks: join text, skip thinking, error on no text --
# Each builder turns ("text" | "thinking", str) specs into a real SDK response.

Blocks = Optional[list[tuple[str, str]]]


def _cohere(blocks: Blocks) -> Any:
    cohere = pytest.importorskip("cohere", minversion="5.17.0")

    def block(kind: str, text: str) -> Any:
        if kind == "text":
            return cohere.TextAssistantMessageResponseContentItem(text=text)
        model = cohere.ThinkingAssistantMessageResponseContentItem
        field = "thinking" if "thinking" in model.model_fields else "value"
        return model.model_validate({field: text})

    content = None if blocks is None else [block(*spec) for spec in blocks]
    return cohere.ChatResponse(
        id="c",
        finish_reason="COMPLETE",
        message=cohere.AssistantMessageResponse(content=content),
    )


def _openai_compatible(blocks: Any) -> Any:
    if isinstance(blocks, list):
        blocks = [
            {"type": "text" if kind == "text" else "reasoning", "text": text}
            if isinstance(kind, str)
            else kind
            for kind, text in blocks
        ]
    response = chat_completion()
    # The SDK's permissive response construction retains provider-specific lists.
    response.choices[0].message = ChatCompletionMessage.model_construct(
        role="assistant", content=blocks
    )
    return response


def _mistral(blocks: Blocks) -> Any:
    m = pytest.importorskip("mistralai.client.models")
    content = (
        None
        if blocks is None
        else [
            m.TextChunk(text=text)
            if kind == "text"
            else m.ThinkChunk(thinking=[m.TextChunk(text=text)])
            for kind, text in blocks
        ]
    )
    return m.ChatCompletionResponse(
        id="r",
        object="chat.completion",
        model="magistral",
        created=1,
        usage=m.UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        choices=[
            m.ChatCompletionChoice(
                index=0,
                finish_reason="stop",
                message=m.AssistantMessage(content=content),
            )
        ],
    )


class BlockProvider(NamedTuple):
    provider: Provider
    build: Callable[[Any], Any]
    no_text_error: type[Exception]


BLOCK_PROVIDERS = {
    "cohere": BlockProvider(Provider.COHERE, _cohere, ResponseParsingError),
    "databricks": BlockProvider(
        Provider.DATABRICKS, _openai_compatible, ValidationError
    ),
    "mistral": BlockProvider(Provider.MISTRAL, _mistral, ValidationError),
}
BLOCK_MODES = [
    ("cohere", Mode.TOOLS),
    ("cohere", Mode.JSON_SCHEMA),
    ("cohere", Mode.MD_JSON),
    ("databricks", Mode.JSON),
    ("databricks", Mode.JSON_SCHEMA),
    ("databricks", Mode.MD_JSON),
    ("mistral", Mode.JSON_SCHEMA),
    ("mistral", Mode.MD_JSON),
]
# Fenced JSON split across blocks needs a mode that strips Markdown fences.
FENCED = {("cohere", Mode.TOOLS), ("cohere", Mode.MD_JSON)} | {
    (name, Mode.MD_JSON) for name in BLOCK_PROVIDERS
}
TEXT_BLOCKS: dict[str, list[tuple[str, str]]] = {
    "single": [("text", '{"name":"Ada"}')],
    "split": [("text", '{"name":'), ("text", '"Ada"}')],
    "empty-first": [("text", ""), ("text", '{"name":"Ada"}')],
    "three-way": [("text", '{"na'), ("text", 'me":"A'), ("text", 'da"}')],
    "thinking-between": [
        ("thinking", 'draft: {"name": "x"}'),
        ("text", '{"name":'),
        ("thinking", "still not JSON"),
        ("text", '"Ada"}'),
    ],
    "fence-spans-blocks": [
        ("text", "```json\n"),
        ("text", '{"name":"Ada"}'),
        ("text", "\n```"),
    ],
}
NO_TEXT: dict[str, Blocks] = {
    "null": None,
    "empty-list": [],
    "only-thinking": [("thinking", "reasoning")],
}


def _parse(name: str, mode: Mode, response: Any) -> Any:
    provider = BLOCK_PROVIDERS[name].provider
    return mode_registry.get_handlers(provider, mode).response_parser(response, Item)


@pytest.mark.parametrize(
    ("name", "mode", "blocks"),
    [
        pytest.param(name, mode, blocks, id=f"{name}-{mode.value}-{label}")
        for name, mode in BLOCK_MODES
        for label, blocks in TEXT_BLOCKS.items()
        if label != "fence-spans-blocks" or (name, mode) in FENCED
    ],
)
def test_text_blocks_are_joined_and_thinking_skipped(
    name: str, mode: Mode, blocks: list[tuple[str, str]]
) -> None:
    response = BLOCK_PROVIDERS[name].build(blocks)
    original = response.model_copy(deep=True)

    result = _parse(name, mode, response)

    assert result == Item(name="Ada")
    assert response == original


@pytest.mark.parametrize("blocks", NO_TEXT.values(), ids=NO_TEXT.keys())
@pytest.mark.parametrize(("name", "mode"), BLOCK_MODES, ids=str)
def test_response_without_text_blocks_is_an_error(
    name: str, mode: Mode, blocks: Blocks
) -> None:
    provider = BLOCK_PROVIDERS[name]
    response = provider.build(blocks)

    with pytest.raises(provider.no_text_error) as exc_info:
        _parse(name, mode, response)

    if isinstance(exc_info.value, ResponseParsingError):
        assert exc_info.value.raw_response is response


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON])
def test_cohere_v1_single_text_response_is_supported(mode: Mode) -> None:
    cohere = pytest.importorskip("cohere", minversion="5.17.0")
    response = cohere.NonStreamedChatResponse(
        text='{"name":"Ada"}', finish_reason="COMPLETE"
    )
    assert _parse("cohere", mode, response) == Item(name="Ada")


@pytest.mark.parametrize("mode", [Mode.JSON, Mode.JSON_SCHEMA, Mode.MD_JSON])
@pytest.mark.parametrize(
    ("content", "ok"),
    [
        pytest.param('{"name":"Ada"}', True, id="string"),
        pytest.param(
            [{"type": "text", "text": '{"name":"Ada"}', "thoughtSignature": "sig"}],
            True,
            id="text-block-with-metadata",
        ),
        pytest.param("", False, id="empty-string"),
        pytest.param(
            [None, {"type": "text"}, {"type": "text", "text": 1}],
            False,
            id="invalid-blocks",
        ),
    ],
)
def test_openai_compatible_raw_content_shapes(
    mode: Mode, content: Any, ok: bool
) -> None:
    response = chat_completion()
    response.choices[0].message = ChatCompletionMessage.model_construct(
        role="assistant", content=content
    )
    parse = mode_registry.get_handlers(Provider.DATABRICKS, mode).response_parser
    if not ok:
        with pytest.raises(ValidationError):
            parse(response, Item)
        return
    result = parse(response, Item, strict=True)
    assert result == Item(name="Ada")
    assert getattr(result, "_raw_response", None) is response


def test_databricks_md_json_text_blocks_through_sdk() -> None:
    content = [
        {
            "type": "text",
            "text": '```json\n{"tasks":[{"name":"Ada"}]}\n```',
            "thoughtSignature": "sig",
        }
    ]
    payload = chat_completion().model_dump()
    payload["choices"][0]["message"]["content"] = content

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/serving-endpoints/chat/completions"
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = instructor.from_provider(
            "databricks/databricks-gemini-3-5-flash",
            mode=Mode.MD_JSON,
            api_key="test-key",
            base_url="https://databricks.example",
            http_client=http_client,
        )
        result = client.create(
            response_model=list[Item],
            messages=[{"role": "user", "content": "Extract Ada"}],
            max_retries=1,
        )

    assert result == [Item(name="Ada")]
