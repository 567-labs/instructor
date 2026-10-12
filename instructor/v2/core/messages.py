"""Message helpers owned by the v2 runtime."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam


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

    message_count = len(messages)
    new_messages: list[dict[str, Any]] = []
    flat_string = True
    for message in messages[: min(10, message_count)]:
        if not isinstance(message.get("content", ""), str):
            flat_string = False
            break

    if flat_string and message_count > 10:
        flat_string = all(
            isinstance(message.get("content", ""), str) for message in messages[10:]
        )

    for message in messages:
        role = message.get("role", "user")
        new_content = message.get("content", "")
        is_plain_message = set(message).issubset({"role", "content"})
        if new_content is None:
            new_content = ""
        elif isinstance(new_content, list):
            new_content = list(new_content)
        if not flat_string and isinstance(new_content, str):
            new_content = [{"type": "text", "text": new_content}]

        if (
            new_messages
            and role == new_messages[-1]["role"]
            and is_plain_message
            and set(new_messages[-1]).issubset({"role", "content"})
        ):
            if flat_string:
                new_messages[-1]["content"] += f"\n\n{new_content}"
            elif isinstance(new_content, list):
                new_messages[-1]["content"].extend(new_content)
            else:
                new_messages[-1]["content"].append(new_content)
        else:
            new_message = dict(message)
            new_message.update(role=role, content=new_content)
            new_messages.append(new_message)

    return new_messages


def get_message_content(message: ChatCompletionMessageParam) -> list[Any]:
    """Return message content in list form for Gemini-style APIs.

    The returned list is always a new object: callers treat it as their own
    `parts` list (e.g. `transform_to_gemini_prompt` hoists the system prompt
    into it with `.insert(0, ...)`), and returning the caller's own content
    list would splice the hoisted prompt into the caller's message.
    """
    if not message:
        return [""]
    content = message.get("content", "")
    if isinstance(content, list):
        return list(content) if content else [""]
    return [content if content is not None else ""]
