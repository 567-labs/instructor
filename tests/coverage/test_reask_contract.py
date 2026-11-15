"""Contract: reask handlers append feedback for any failed response shape.

The retry loop hands the failed raw response to the provider's reask handler.
That response may be missing (`None`), an unconsumed stream (issue #1991), a
gateway reply with empty `choices`, a reply without a tool call, or a Responses
API reply with reasoning items. Every handler must append the correction
without crashing or mutating the caller's messages. Responses are real SDK
objects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response
from pydantic import BaseModel, ValidationError, field_validator

# Importing the modules registers the lazy handlers.
import instructor.v2.providers.openai.handlers  # noqa: F401
import instructor.v2.providers.perplexity.handlers  # noqa: F401
from instructor.processing.response import handle_reask_kwargs
from instructor.v2.core.errors import InstructorRetryException, ResponseParsingError
from instructor.v2.core.messages import isolate_retry_kwargs
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.core.retry import retry_sync_v2
from tests.coverage._openai import chat_completion


class Answer(BaseModel):
    name: str


def _validation_error() -> ValidationError:
    class Named(BaseModel):
        name: str

        @field_validator("name")
        @classmethod
        def must_have_space(cls, value: str) -> str:
            if " " not in value:
                raise ValueError("must contain space")
            return value

    try:
        Named(name="John")
    except ValidationError as error:
        return error
    raise AssertionError("validation must fail")


def _openai_stream() -> openai.Stream[ChatCompletionChunk]:
    return openai.Stream(
        cast_to=ChatCompletionChunk,
        response=httpx.Response(200, content=b""),
        client=openai.OpenAI(api_key="local"),
    )


def _mistral(choices: list[dict[str, Any]]) -> Any:
    models = pytest.importorskip("mistralai.client.models")
    return models.ChatCompletionResponse.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "model": "m",
            "created": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "choices": choices,
        }
    )


def _mistral_message(message: dict[str, Any]) -> Any:
    return _mistral(
        [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", **message},
            }
        ]
    )


def _anthropic_text() -> Any:
    types = pytest.importorskip("anthropic.types")
    return types.Message(
        id="m",
        type="message",
        role="assistant",
        model="m",
        content=[types.TextBlock(type="text", text="no tool call")],
        stop_reason="end_turn",
        usage=types.Usage(input_tokens=0, output_tokens=0),
    )


def _empty_choices(provider: Provider) -> Any:
    if provider is Provider.MISTRAL:
        return _mistral([])
    if provider is Provider.BEDROCK:
        return {"output": {}, "stopReason": "end_turn"}
    return ChatCompletion(
        id="c", choices=[], created=0, model="m", object="chat.completion"
    )


def _text_only(provider: Provider) -> Any:
    if provider is Provider.MISTRAL:
        return _mistral_message({"content": "no tool call"})
    if provider is Provider.ANTHROPIC:
        return _anthropic_text()
    if provider is Provider.BEDROCK:
        text = [{"text": "no tool call"}]
        return {"output": {"message": {"role": "assistant", "content": text}}}
    return chat_completion(content="no tool call")


VARIANTS: dict[str, Callable[[Provider], Any]] = {
    "none": lambda _provider: None,
    "stream": lambda _provider: _openai_stream(),
    "empty-choices": _empty_choices,
    "no-tool-call": _text_only,
}
NO_REPLAY = ["user", "user"]
REPLAY = ["user", "assistant", "user"]
CASES: list[tuple[Provider, Mode, list[str]]] = [
    # (provider, mode, roles after reasking a text-only reply)
    (Provider.OPENAI, Mode.TOOLS, REPLAY),
    (Provider.OPENAI, Mode.JSON, REPLAY),
    (Provider.OPENAI, Mode.MD_JSON, REPLAY),
    (Provider.OPENAI, Mode.RESPONSES_TOOLS, NO_REPLAY),
    (Provider.PERPLEXITY, Mode.MD_JSON, REPLAY),
    (Provider.MISTRAL, Mode.TOOLS, REPLAY),
    (Provider.MISTRAL, Mode.JSON_SCHEMA, REPLAY),
    (Provider.MISTRAL, Mode.MD_JSON, REPLAY),
    (Provider.ANTHROPIC, Mode.TOOLS, REPLAY),
    (Provider.ANTHROPIC, Mode.JSON, NO_REPLAY),
    (Provider.ANTHROPIC, Mode.JSON_SCHEMA, NO_REPLAY),
    (Provider.BEDROCK, Mode.TOOLS, REPLAY),
    (Provider.BEDROCK, Mode.TOOLS_STRICT, REPLAY),
    (Provider.BEDROCK, Mode.MD_JSON, REPLAY),
    (Provider.BEDROCK, Mode.JSON_SCHEMA, REPLAY),
    (Provider.COHERE, Mode.TOOLS, NO_REPLAY),
]
MATRIX = [
    pytest.param(
        provider,
        mode,
        variant,
        text_roles if variant == "no-tool-call" else NO_REPLAY,
        id=f"{provider.value}-{mode.value}-{variant}",
    )
    for provider, mode, text_roles in CASES
    for variant in VARIANTS
]


@pytest.mark.parametrize(("provider", "mode", "variant", "roles"), MATRIX)
def test_reask_appends_feedback_for_any_response_shape(
    provider: Provider, mode: Mode, variant: str, roles: list[str]
) -> None:
    caller_messages = [{"role": "user", "content": "Ada is 37 years old"}]
    kwargs = isolate_retry_kwargs({"model": "m", "messages": caller_messages})

    reasked = mode_registry.get_handlers(provider, mode).reask_handler(
        kwargs=kwargs,
        response=VARIANTS[variant](provider),
        exception=_validation_error(),
    )

    assert caller_messages == [{"role": "user", "content": "Ada is 37 years old"}]
    assert [message["role"] for message in reasked["messages"]] == roles


@pytest.mark.filterwarnings("ignore:Mode.ANTHROPIC_TOOLS:DeprecationWarning")
@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.ANTHROPIC_TOOLS, Mode.MD_JSON])
def test_compat_dispatcher_reasks_an_unconsumed_stream(mode: Mode) -> None:
    result = handle_reask_kwargs(
        kwargs={"messages": [{"role": "user", "content": "test"}]},
        mode=mode,
        response=_openai_stream(),
        exception=_validation_error(),
    )
    assert len(result["messages"]) == 2


def test_responses_reask_skips_reasoning_items_and_names_the_tool_call() -> None:
    response = Response.model_validate(
        {
            "id": "r",
            "created_at": 0,
            "object": "response",
            "model": "m",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {"type": "reasoning", "id": "rs", "summary": []},
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "extract_person",
                    "arguments": '{"name": "Jane"}',
                },
            ],
        }
    )
    result = mode_registry.get_handlers(
        Provider.OPENAI, Mode.RESPONSES_TOOLS
    ).reask_handler(
        kwargs={"messages": [{"role": "user", "content": "test"}]},
        response=response,
        exception=_validation_error(),
    )

    assert len(result["messages"]) == 2
    feedback = result["messages"][-1]["content"]
    assert "tool call name=extract_person, id=call_123" in feedback
    assert '{"name": "Jane"}' in feedback


def test_mistral_reask_replays_a_usable_tool_call() -> None:
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "Answer", "arguments": '{"wrong": true}'},
    }
    reasked = mode_registry.get_handlers(Provider.MISTRAL, Mode.TOOLS).reask_handler(
        kwargs=isolate_retry_kwargs({"model": "m", "messages": [{"role": "user"}]}),
        response=_mistral_message({"content": None, "tool_calls": [call]}),
        exception=ValueError("boom"),
    )
    assert [message["role"] for message in reasked["messages"]] == [
        "user",
        "assistant",
        "tool",
    ]


# --- GenAI: contents, function-call replay, thought signatures --------------


def _genai_response(*parts: Any) -> Any:
    types = pytest.importorskip("google.genai.types")
    content = types.Content(role="model", parts=list(parts))
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.parametrize("variant", ["none", "stream", "empty-choices"])
def test_genai_reask_without_candidates_appends_only_feedback(
    mode: Mode, variant: str
) -> None:
    pytest.importorskip("google.genai.types")
    reasked = mode_registry.get_handlers(Provider.GENAI, mode).reask_handler(
        kwargs={},
        response=VARIANTS[variant](Provider.GENAI),
        exception=_validation_error(),
    )
    assert [content.role for content in reasked["contents"]] == ["user"]


@pytest.mark.parametrize("call_first", [True, False])
def test_genai_tools_reask_replays_function_call_with_signature(
    call_first: bool,
) -> None:
    types = pytest.importorskip("google.genai.types")
    call = types.Part.from_function_call(name="test_fn", args={"value": 1})
    call.thought_signature = b"sig"
    preface = types.Part.from_text(text="some preface")
    response = _genai_response(*([call] if call_first else [preface, call]))
    model_content = response.candidates[0].content
    original: dict[str, Any] = {"contents": []}

    result = mode_registry.get_handlers(Provider.GENAI, Mode.TOOLS).reask_handler(
        kwargs=original, response=response, exception=Exception("boom")
    )

    assert original["contents"] == []
    assert result["contents"][-2] is model_content
    assert model_content.parts[-1].thought_signature == b"sig"
    assert result["contents"][-1].role == "user"
    assert result["contents"][-1].parts[0].function_response.name == "test_fn"


def test_genai_tools_reask_without_function_call_replays_model_text() -> None:
    types = pytest.importorskip("google.genai.types")
    response = _genai_response(types.Part.from_text(text="not a function call"))

    result = mode_registry.get_handlers(Provider.GENAI, Mode.TOOLS).reask_handler(
        kwargs={"contents": []}, response=response, exception=Exception("boom")
    )

    assert result["contents"][0] is response.candidates[0].content
    assert result["contents"][1].role == "user"


# --- Vertex AI: contents built from GenerationResponse candidates -----------


def _vertexai_response(provider: Provider, variant: str) -> Any:
    if variant in ("none", "stream"):
        return VARIANTS[variant](provider)
    gm = pytest.importorskip("vertexai.generative_models")
    if variant == "no-tool-call":
        content = {"role": "model", "parts": [{"text": "no tool call"}]}
        return gm.GenerationResponse.from_dict({"candidates": [{"content": content}]})
    # A safety-blocked candidate carries no content parts to replay.
    blocked = {"finish_reason": "SAFETY"}
    return gm.GenerationResponse.from_dict({"candidates": [blocked]})


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.PARALLEL_TOOLS, Mode.MD_JSON])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_vertexai_reask_appends_feedback_for_any_response_shape(
    mode: Mode, variant: str
) -> None:
    pytest.importorskip("vertexai.generative_models")
    caller_contents: list[Any] = []
    reasked = mode_registry.get_handlers(Provider.VERTEXAI, mode).reask_handler(
        kwargs={"contents": caller_contents},
        response=_vertexai_response(Provider.VERTEXAI, variant),
        exception=_validation_error(),
    )
    assert caller_contents == []
    roles = ["model", "user"] if variant == "no-tool-call" else ["user"]
    assert [content.role for content in reasked["contents"]] == roles


# --- retry loop --------------------------------------------------------------


@pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.MISTRAL])
def test_empty_choices_response_retries_instead_of_crashing(provider: Provider) -> None:
    calls = 0

    def create(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return _empty_choices(provider)

    with pytest.raises(InstructorRetryException) as excinfo:
        retry_sync_v2(
            func=create,
            response_model=Answer,
            provider=provider,
            mode=Mode.TOOLS,
            context=None,
            max_retries=1,
            args=(),
            kwargs={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            strict=True,
        )

    # Parsed like any other failure; never an IndexError from the reask handler.
    assert isinstance(excinfo.value.__cause__, ResponseParsingError)
    assert calls == excinfo.value.n_attempts == 2
    assert len(excinfo.value.failed_attempts or []) == 2
