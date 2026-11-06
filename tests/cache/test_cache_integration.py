import types

import instructor
from instructor.cache import AutoCache
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, field_validator  # type: ignore[import-not-found]


def test_auto_cache_prevents_duplicate_provider_calls(monkeypatch):
    _ = monkeypatch  # unused fixture for parity with other tests
    """Ensure that AutoCache prevents duplicate provider calls via patch layer."""

    class User(BaseModel):
        name: str = Field(...)

    call_counter = {"n": 0}

    # Fake provider completion function mimicking minimal OpenAI chat response
    def fake_completion(*_args, **_kwargs):  # noqa: D401, ANN001
        call_counter["n"] += 1
        content = User(name="cached").model_dump_json()
        # Return minimal ChatCompletion-like object
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content),
                    finish_reason="stop",
                )
            ],
            usage={},
        )

    # Create Instructor client using from_litellm so we go through patch stack
    cache = AutoCache(maxsize=10)
    client = instructor.from_litellm(fake_completion, mode=instructor.Mode.JSON)

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hello"}]

    # First call – provider should be invoked
    _ = client.create(messages=list(messages), response_model=User, cache=cache)
    assert call_counter["n"] == 1

    # Second call with identical inputs – should hit cache, no new provider call
    _ = client.create(messages=list(messages), response_model=User, cache=cache)
    assert call_counter["n"] == 1, "Cache miss – provider was called again"


def test_auto_cache_prevents_duplicate_calls_after_a_retry(monkeypatch):
    _ = monkeypatch
    """Regression test: a call that needed a retry must still be cacheable.

    Reask handlers append/extend the request's messages list in place. If that
    list is the same object patch.py reads again to compute the cache store key,
    the store key ends up different from the lookup key computed before the
    retry, so a later, identical call never hits the cache.
    """

    class Answer(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def must_be_positive(cls, v: int) -> int:
            if v < 0:
                raise ValueError("value must be positive")
            return v

    call_counter = {"n": 0}

    def fake_completion(*_args, **_kwargs):
        call_counter["n"] += 1
        # First call ever returns an invalid value, forcing exactly one retry.
        value = -5 if call_counter["n"] == 1 else 42
        content = Answer.model_construct(value=value).model_dump_json()
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content),
                    finish_reason="stop",
                )
            ],
            usage={},
        )

    cache = AutoCache(maxsize=10)
    client = instructor.from_litellm(fake_completion, mode=instructor.Mode.JSON)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "what is 6 times 7?"}
    ]

    result1 = client.create(
        messages=list(messages), response_model=Answer, max_retries=2, cache=cache
    )
    assert result1.value == 42
    assert call_counter["n"] == 2, "First call should need exactly one retry"

    result2 = client.create(
        messages=list(messages), response_model=Answer, max_retries=2, cache=cache
    )
    assert result2.value == 42
    assert call_counter["n"] == 2, (
        "Second, identical call should hit the cache instead of calling the provider again"
    )


def test_cache_key_accounts_for_hoisted_system_prompt():
    """Regression test: a system prompt hoisted out of ``messages`` must be hashed.

    Providers like Anthropic and Bedrock move ``role="system"`` messages out of
    ``messages`` into a separate top-level ``system`` parameter during request
    preparation, which happens *before* patch.py computes the cache key. If the
    key only hashes the remaining ``messages``, two calls that differ solely in
    their system prompt collide and the second silently returns the first
    call's answer.
    """
    import anthropic
    from anthropic.types import Message, TextBlock, Usage

    class Answer(BaseModel):
        value: str

    call_counter = {"n": 0}

    def fake_create(*_args, **_kwargs):
        call_counter["n"] += 1
        content = Answer(value=f"call-{call_counter['n']}").model_dump_json()
        return Message(
            id="msg_1",
            model="claude-3-5-sonnet-latest",
            role="assistant",
            type="message",
            stop_reason="end_turn",
            content=[TextBlock(type="text", text=content)],
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    raw_client = anthropic.Anthropic(api_key="sk-not-used")
    raw_client.messages.create = fake_create  # ty: ignore[invalid-assignment]
    client = instructor.from_anthropic(raw_client, mode=instructor.Mode.JSON)

    cache = AutoCache(maxsize=10)

    def ask(system_prompt: str) -> Answer:
        return client.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=100,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "hello"},
            ],
            response_model=Answer,
            cache=cache,
        )

    first = ask("Always answer in French.")
    assert call_counter["n"] == 1
    assert first.value == "call-1"

    second = ask("Always answer in German.")
    assert call_counter["n"] == 2, (
        "A different system prompt must not reuse the cached response"
    )
    assert second.value == "call-2"

    # Repeating the very first request must still hit the cache.
    repeat = ask("Always answer in French.")
    assert call_counter["n"] == 2, "Identical request should still be served from cache"
    assert repeat.value == "call-1"
