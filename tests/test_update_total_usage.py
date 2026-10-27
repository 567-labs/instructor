from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from instructor.utils.core import update_total_usage


class ProviderSpecificUsage(CompletionUsage):
    def get(self, key: str, default=None):
        return getattr(self, key, default)


def make_response(usage: CompletionUsage) -> ChatCompletion:
    return ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                logprobs=None,
                message=ChatCompletionMessage(role="assistant", content="ok"),
            )
        ],
        created=123,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=usage,
    )


def test_update_total_usage_preserves_openai_usage_subclass() -> None:
    response = make_response(
        ProviderSpecificUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    )
    total_usage = CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)

    updated = update_total_usage(response, total_usage)

    assert updated is not None
    assert isinstance(updated.usage, ProviderSpecificUsage)
    assert updated.usage.get("prompt_tokens") == 16
    assert updated.usage.get("completion_tokens") == 10
    assert updated.usage.total_tokens == 26


def test_usage_helpers_gracefully_handle_missing_openai() -> None:
    import sys
    from typing import Any
    from unittest.mock import patch
    from instructor.v2.core.usage import has_compatible_usage, update_total_usage
    from instructor.v2.core.retry import _initialize_usage
    from instructor.v2.core.providers import Provider

    class DummyResponse:
        def __init__(self, usage=None):
            self.usage = usage

    with patch.dict(
        sys.modules,
        {
            "openai": None,
            "openai.types": None,
            "openai.types.completion_usage": None,
        },
    ):
        dummy_total: Any = object()
        assert not has_compatible_usage(DummyResponse(), dummy_total)
        res = update_total_usage(DummyResponse(), dummy_total)
        assert res is not None
        assert _initialize_usage(Provider.OPENAI) is None

