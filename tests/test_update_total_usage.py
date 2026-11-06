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
