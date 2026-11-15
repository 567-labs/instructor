"""Message helpers owned by the v2 runtime."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam


def text_from_content_blocks(blocks: list[Any]) -> str | None:
    """Join text fragments while leaving opaque provider blocks untouched."""
    parts = []
    for block in blocks:
        if isinstance(block, dict):
            kind, text = block.get("type"), block.get("text")
        else:
            kind, text = getattr(block, "type", None), getattr(block, "text", None)
        if kind == "text" and isinstance(text, str):
            parts.append(text)
    return "".join(parts) if parts else None


def append_system_instruction(message: dict[str, Any], instruction: str) -> None:
    """Append `instruction` to a mutable system message's first text part."""
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = f"{content}\n\n{instruction}"
        return
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part["text"] += f"\n\n{instruction}"
                return
        content.append({"type": "text", "text": instruction})
        return
    message["content"] = instruction


def extract_messages(kwargs: dict[str, Any]) -> Any:
    if "messages" in kwargs:
        return kwargs["messages"]
    if "contents" in kwargs:
        return kwargs["contents"]
    if "chat_history" in kwargs:
        return kwargs["chat_history"]
    return []


def copy_messages_for_mutation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of `messages` that is safe to mutate in place.

    Handlers that inject a system/instruction message do so by mutating a
    `messages` list and its dict elements directly (`.insert()`, `.append()`,
    `message["content"] += ...`). Since `kwargs.copy()` is shallow, `messages`
    is otherwise the exact list (and dict) objects the caller passed in, so
    those in-place edits would corrupt the caller's own conversation state.
    """
    copied = [dict(message) for message in messages]
    for message in copied:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                dict(item) if isinstance(item, dict) else item for item in content
            ]
    return copied


def add_json_schema_prompt(
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    *,
    codeblock: bool = False,
    suffix: str = "",
    merge: bool = True,
) -> list[dict[str, Any]]:
    """Return a copy of `messages` with the JSON-schema system instruction."""
    instruction = dedent(
        f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n

            {json.dumps(schema, indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself{suffix}
            """
    )
    messages = copy_messages_for_mutation(messages)
    if messages and messages[0]["role"] == "system":
        append_system_instruction(messages[0], instruction)
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    if codeblock:
        messages.append(
            {
                "role": "user",
                "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
            }
        )
    return merge_consecutive_messages(messages) if merge else messages


def first_choice_message(response: Any) -> Any | None:
    """Return `choices[0].message`, or None for streams and empty `choices`."""
    choices = getattr(response, "choices", None)
    return choices[0].message if choices else None


def isolate_retry_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Copy request lists that reask handlers mutate during retries."""
    isolated = dict(kwargs)
    for key_name in ("messages", "contents", "chat_history"):
        value = isolated.get(key_name)
        if isinstance(value, list):
            isolated[key_name] = list(value)
    return isolated


def dump_message(message: ChatCompletionMessage) -> ChatCompletionMessageParam:
    ret: ChatCompletionMessageParam = {
        "role": message.role,
        "content": message.content or "",
    }
    if hasattr(message, "tool_calls") and message.tool_calls is not None:
        ret["tool_calls"] = message.model_dump()["tool_calls"]
    if hasattr(message, "function_call") and message.function_call is not None:
        if not isinstance(ret["content"], str):
            response_message = ""
            for content_message in ret["content"]:
                if isinstance(content_message, dict):
                    if content_message.get("type") == "text":
                        response_message += content_message.get("text", "")
                    elif content_message.get("type") == "refusal":
                        response_message += content_message.get("refusal", "")
            ret["content"] = response_message
        ret["content"] += json.dumps(message.model_dump()["function_call"])
    return ret


def merge_consecutive_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent plain messages without crossing protocol boundaries."""
    if not messages:
        return []

    new_messages: list[dict[str, Any]] = []
    flat_string = all(
        isinstance(message.get("content", ""), str) for message in messages
    )
    for message in messages:
        role = message.get("role", "user")
        new_content = message.get("content", "")
        if new_content is None:
            new_content = ""
        if flat_string:
            new_content = [new_content]
        elif isinstance(new_content, str):
            new_content = [{"type": "text", "text": new_content}]
        elif isinstance(new_content, list):
            new_content = list(new_content)

        if (
            new_messages
            and role == new_messages[-1]["role"]
            and set(message).issubset({"role", "content"})
            and set(new_messages[-1]).issubset({"role", "content"})
        ):
            if isinstance(new_content, list):
                new_messages[-1]["content"].extend(new_content)
            else:
                new_messages[-1]["content"].append(new_content)
        else:
            new_message = dict(message)
            new_message.update(role=role, content=new_content)
            new_messages.append(new_message)

    if flat_string:
        for message in new_messages:
            message["content"] = "\n\n".join(message["content"])
    return new_messages


def get_message_content(message: ChatCompletionMessageParam | None) -> list[Any]:
    """Return a new list of message content for Gemini-style APIs."""
    if not message:
        return [""]
    content = message.get("content", "")
    if isinstance(content, list):
        return list(content) if content else [""]
    return [content if content is not None else ""]
