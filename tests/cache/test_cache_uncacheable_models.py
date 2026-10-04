"""A `cache=` that silently never fills.

`load_cached_response` rebuilds a cached value with
`response_model.model_validate_json`, so only a Pydantic result can be stored.
`list[User]`, `Iterable[User]` and `str` come back from the retry loop as a
`ListResponse` or a plain scalar, and the old store path skipped them without a
word: the call succeeded, the cache stayed empty, and every request went to the
provider again. These tests pin the warning that says so.
"""

import json
import logging

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from instructor.cache import AutoCache
from instructor.v2.core.mode import Mode
from instructor.v2.core.patch import patch
from instructor.v2.core.providers import Provider

LOGGER_NAME = "instructor.v2"


class User(BaseModel):
    name: str
    age: int


def _completion(payload: dict, tool_name: str = "User") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-fake",
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(payload),
                            },
                        }
                    ],
                ),
            )
        ],
        usage=CompletionUsage(completion_tokens=5, prompt_tokens=10, total_tokens=15),
    )


def _create(payload: dict, tool_name: str = "User"):
    calls: list[int] = []

    def fake_create(*_args, **_kwargs):
        calls.append(1)
        return _completion(payload, tool_name=tool_name)

    return patch(create=fake_create, mode=Mode.TOOLS, provider=Provider.OPENAI), calls


def _messages() -> list[dict]:
    return [{"role": "user", "content": "hi"}]


def test_cache_warns_when_a_list_response_model_cannot_be_stored(caplog):
    create, calls = _create({"tasks": [{"name": "Alice", "age": 30}]})
    cache = AutoCache(maxsize=10)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        first = create(messages=_messages(), response_model=list[User], cache=cache)
        second = create(messages=_messages(), response_model=list[User], cache=cache)

    assert [u.name for u in first] == ["Alice"]
    assert [u.name for u in second] == ["Alice"]
    # The list result is not storable, so the provider is still called twice.
    assert len(calls) == 2
    assert len(cache._cache) == 0

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    # str(list[User]) is module-qualified inside the test module, so match the
    # shape rather than the exact name.
    assert any("list[" in message and "User]" in message for message in warnings), (
        warnings
    )
    assert any("ListResponse" in message for message in warnings), warnings


def test_cache_warns_when_a_scalar_response_model_cannot_be_stored(caplog):
    create, calls = _create({"content": "hello"}, tool_name="Response")
    cache = AutoCache(maxsize=10)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        create(messages=_messages(), response_model=str, cache=cache)

    assert len(calls) == 1
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("str" in message for message in warnings), warnings


def test_cache_still_stores_a_pydantic_result_without_warning(caplog):
    create, calls = _create({"name": "Alice", "age": 30})
    cache = AutoCache(maxsize=10)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        first = create(messages=_messages(), response_model=User, cache=cache)
        second = create(messages=_messages(), response_model=User, cache=cache)

    assert first.name == second.name == "Alice"
    assert len(calls) == 1, "the second call should have come from the cache"
    assert len(cache._cache) == 1

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("Caching is not supported" in message for message in warnings)
