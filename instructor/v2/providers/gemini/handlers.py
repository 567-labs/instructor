"""Gemini v2 mode handlers."""

from __future__ import annotations

import inspect
import json
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterable as TypingIterable,
)
from typing import Any, cast

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.partial import PartialBase
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.providers.gemini.utils import (
    extract_gemini_chunk_text,
    handle_gemini_json,
    handle_gemini_tools,
)
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.errors import ResponseParsingError
from instructor.v2.core.json import extract_json_from_codeblock


def reask_gemini_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Build a Gemini tool reask payload after validation failure."""
    from google.ai import generativelanguage as glm  # type: ignore

    function_call = None
    for part in getattr(response, "parts", None) or []:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            break

    if function_call is None:
        # The model answered in prose instead of calling the tool, so there
        # is no function_call to re-attach. Mirror the sibling guards
        # (openai/genai/mistral reask handlers): fall back to a plain user
        # correction instead of iterating None.
        prose = getattr(response, "text", "") or ""
        reask_msgs = []
        if prose:
            reask_msgs.append({"role": "model", "parts": [prose]})
        reask_msgs.append(
            {
                "role": "user",
                "parts": [
                    f"Validation Error(s) found:\n{exception}\nRecall the function arguments correctly and fix the errors"
                ],
            }
        )
        kwargs["contents"].extend(reask_msgs)
        return kwargs

    reask_msgs = [
        {
            "role": "model",
            "parts": [
                glm.FunctionCall(
                    name=response.parts[0].function_call.name,
                    args=response.parts[0].function_call.args,
                )
            ],
        },
        {
            "role": "function",
            "parts": [
                glm.Part(
                    function_response=glm.FunctionResponse(
                        name=response.parts[0].function_call.name,
                        response={"error": f"Validation Error(s) found:\n{exception}"},
                    )
                ),
            ],
        },
        {
            "role": "user",
            "parts": ["Recall the function arguments correctly and fix the errors"],
        },
    ]
    kwargs["contents"].extend(reask_msgs)
    return kwargs


def reask_gemini_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Build a Gemini JSON reask payload after validation failure."""
    try:
        text = response.text
    except ValueError:
        text = None
    if text is None:
        text = "<no readable text in the previous response>"
    kwargs["contents"].append(
        {
            "role": "user",
            "parts": [
                "Correct the following JSON response, based on the errors given below:\n\n"
                f"JSON:\n{text}\n\nExceptions:\n{exception}"
            ],
        }
    )
    return kwargs


def parse_gemini_json(
    response_model: type[BaseModel],
    completion: Any,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
) -> BaseModel:
    """Parse Gemini text JSON responses."""
    try:
        text = completion.text
    except ValueError:
        text = None
    if text is None:
        raise ResponseParsingError(
            "Unable to extract JSON from completion text. The response may have been blocked or empty.",
            mode="GEMINI_JSON",
            raw_response=completion,
        )
    extra_text = extract_json_from_codeblock(text)
    if strict:
        return response_model.model_validate_json(
            extra_text,
            context=validation_context,
            strict=True,
        )
    parsed = json.loads(extra_text, strict=False)
    return response_model.model_validate(
        parsed,
        context=validation_context,
        strict=False,
    )


def parse_gemini_tools(
    response_model: type[BaseModel],
    completion: Any,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
) -> BaseModel:
    """Parse Gemini tool-call responses."""
    try:
        function_call = completion.candidates[0].content.parts[0].function_call
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ResponseParsingError(
            "No tool call found in Gemini response",
            mode="GEMINI_TOOLS",
            raw_response=completion,
        ) from exc
    args = getattr(function_call, "args", None)
    if args is None and hasattr(type(function_call), "to_dict"):
        try:
            resp_dict = type(function_call).to_dict(function_call)
        except (AttributeError, TypeError, ValueError):
            resp_dict = {}
        args = resp_dict.get("args")
    if args is None:
        raise ResponseParsingError(
            "No tool call args found in Gemini response",
            mode="GEMINI_TOOLS",
            raw_response=completion,
        )
    return response_model.model_validate(
        args,
        context=validation_context,
        strict=strict,
    )


class GeminiHandlerBase(ModeHandler):
    """Base handler for Gemini modes."""

    mode: Mode

    def extract_streaming_json(
        self, completion: TypingIterable[Any]
    ) -> Generator[str, None, None]:
        """Extract JSON chunks from Gemini streaming responses."""
        for chunk in completion:
            try:
                if self.mode == Mode.TOOLS:
                    resp = chunk.candidates[0].content.parts[0].function_call
                    resp_dict = type(resp).to_dict(resp)
                    if "args" in resp_dict:
                        yield json.dumps(resp_dict["args"])
                else:
                    yield extract_gemini_chunk_text(chunk)
            except AttributeError:
                continue

    async def extract_streaming_json_async(
        self, completion: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """Extract JSON chunks from Gemini async streams."""
        async for chunk in completion:
            try:
                if self.mode == Mode.TOOLS:
                    resp = chunk.candidates[0].content.parts[0].function_call
                    resp_dict = type(resp).to_dict(resp)
                    if "args" in resp_dict:
                        yield json.dumps(resp_dict["args"])
                else:
                    yield extract_gemini_chunk_text(chunk)
            except AttributeError:
                continue

    def _parse_streaming(
        self,
        response_model: type[BaseModel],
        response: Any,
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> Any:
        parse_kwargs: dict[str, Any] = {}
        if validation_context is not None:
            parse_kwargs["context"] = validation_context
        if strict is not None:
            parse_kwargs["strict"] = strict

        streaming_model = cast(Any, response_model)
        if inspect.isasyncgen(response) or isinstance(response, AsyncIterator):
            return streaming_model.from_streaming_response_async(
                response,
                stream_extractor=self.extract_streaming_json_async,
                **parse_kwargs,
            )

        generator = streaming_model.from_streaming_response(
            response,
            stream_extractor=self.extract_streaming_json,
            **parse_kwargs,
        )
        if inspect.isclass(response_model) and issubclass(response_model, IterableBase):
            return generator
        return list(generator)

    def _finalize(
        self,
        response_model: type[BaseModel],  # noqa: ARG002
        response: Any,
        parsed: Any,  # noqa: ARG002
    ) -> Any:
        if isinstance(parsed, AdapterBase):
            return parsed.content
        if isinstance(parsed, BaseModel):
            parsed._raw_response = response  # type: ignore[attr-defined]
        return parsed


@register_mode_handler(Provider.GEMINI, Mode.TOOLS)
class GeminiToolsHandler(GeminiHandlerBase):
    """Handler for Gemini TOOLS mode."""

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        return handle_gemini_tools(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_gemini_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        if (
            stream
            and inspect.isclass(response_model)
            and issubclass(response_model, (IterableBase, PartialBase))
        ):
            return self._parse_streaming(
                response_model, response, validation_context, strict
            )
        parsed = parse_gemini_tools(
            response_model, response, validation_context, strict
        )
        return self._finalize(response_model, response, parsed)


@register_mode_handler(Provider.GEMINI, Mode.MD_JSON)
class GeminiJSONHandler(GeminiHandlerBase):
    """Handler for Gemini JSON mode."""

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        return handle_gemini_json(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_gemini_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        if (
            stream
            and inspect.isclass(response_model)
            and issubclass(response_model, (IterableBase, PartialBase))
        ):
            return self._parse_streaming(
                response_model, response, validation_context, strict
            )
        parsed = parse_gemini_json(response_model, response, validation_context, strict)
        return self._finalize(response_model, response, parsed)


__all__ = ["GeminiToolsHandler", "GeminiJSONHandler"]
