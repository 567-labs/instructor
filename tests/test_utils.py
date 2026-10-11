import json

import pytest
from instructor.auto_client import supported_providers
from openai.types import CompletionUsage
from pydantic import BaseModel

from instructor.utils import (
    classproperty,
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
    merge_consecutive_messages,
    extract_system_messages,
    combine_system_messages,
    Provider,
    get_provider,
    update_total_usage,
)


def test_extract_json_from_codeblock():
    example = """
    Here is a response

    ```json
    {
        "key": "value"
    }    
    ```
    """
    result = extract_json_from_codeblock(example)
    assert json.loads(result) == {"key": "value"}


def test_extract_json_from_codeblock_no_end():
    example = """
    Here is a response

    ```json
    {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}]
    }  
    """
    result = extract_json_from_codeblock(example)
    assert json.loads(result) == {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}],
    }


def test_extract_json_from_codeblock_no_start():
    example = """
    Here is a response

    {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}, {"key": "value"}]
    }
    """
    result = extract_json_from_codeblock(example)
    assert json.loads(result) == {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}, {"key": "value"}],
    }


def test_stream_json():
    text = """here is the json for you! 
    
    ```json
    , here
    {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}]
    }
    ```

    What do you think?
    """

    def batch_strings(chunks, n=2):
        batch = ""
        for chunk in chunks:
            for char in chunk:
                batch += char
                if len(batch) == n:
                    yield batch
                    batch = ""
        if batch:  # Yield any remaining characters in the last batch
            yield batch

    result = json.loads(
        "".join(list(extract_json_from_stream(batch_strings(text, n=3))))
    )
    assert result == {"key": "value", "another_key": [{"key": {"key": "value"}}]}


@pytest.mark.asyncio
async def test_stream_json_async():
    text = """here is the json for you! 
    
    ```json
    , here
    {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}, {"key": "value"}]
    }
    ```

    What do you think?
    """

    async def batch_strings_async(chunks, n=2):
        batch = ""
        for chunk in chunks:
            for char in chunk:
                batch += char
                if len(batch) == n:
                    yield batch
                    batch = ""
        if batch:  # Yield any remaining characters in the last batch
            yield batch

    result = json.loads(
        "".join(
            [
                chunk
                async for chunk in extract_json_from_stream_async(
                    batch_strings_async(text, n=3)
                )
            ]
        )
    )
    assert result == {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}, {"key": "value"}],
    }


def test_merge_consecutive_messages():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "How are you"},
        {"role": "assistant", "content": "Hello"},
        {"role": "assistant", "content": "I am good"},
    ]
    result = merge_consecutive_messages(messages)
    assert result == [
        {
            "role": "user",
            "content": "Hello\n\nHow are you",
        },
        {
            "role": "assistant",
            "content": "Hello\n\nI am good",
        },
    ]


def test_merge_consecutive_messages_empty():
    messages = []
    result = merge_consecutive_messages(messages)
    assert result == []


def test_merge_consecutive_messages_single():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello"},
    ]
    result = merge_consecutive_messages(messages)
    assert result == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_classproperty():
    """Test custom `classproperty` descriptor."""

    class MyClass:
        @classproperty
        def my_property(cls):
            return cls

    assert MyClass.my_property is MyClass

    class MyClass:
        clvar = 1

        @classproperty
        def my_property(cls):
            return cls.clvar

    assert MyClass.my_property == 1


def test_combine_system_messages_string_string():
    existing = "Existing message"
    new = "New message"
    result = combine_system_messages(existing, new)
    assert result == "Existing message\n\nNew message"


def test_combine_system_messages_list_list():
    existing = [{"type": "text", "text": "Existing"}]
    new = [{"type": "text", "text": "New"}]
    result = combine_system_messages(existing, new)
    assert result == [
        {"type": "text", "text": "Existing"},
        {"type": "text", "text": "New"},
    ]


def test_combine_system_messages_string_list():
    existing = "Existing"
    new = [{"type": "text", "text": "New"}]
    result = combine_system_messages(existing, new)
    assert result == [
        {"type": "text", "text": "Existing"},
        {"type": "text", "text": "New"},
    ]


def test_combine_system_messages_list_string():
    existing = [{"type": "text", "text": "Existing"}]
    new = "New"
    result = combine_system_messages(existing, new)
    assert result == [
        {"type": "text", "text": "Existing"},
        {"type": "text", "text": "New"},
    ]


def test_update_total_usage_preserves_openai_usage_subclass():
    class LiteLLMUsage(CompletionUsage):
        def get(self, key, default=None):
            return getattr(self, key, default)

    class Response(BaseModel):
        usage: LiteLLMUsage

    response_usage = LiteLLMUsage(
        prompt_tokens=3,
        completion_tokens=4,
        total_tokens=7,
    )
    total_usage = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    response = Response(usage=response_usage)

    updated = update_total_usage(response, total_usage)

    assert updated is response
    assert response.usage is response_usage
    assert isinstance(response.usage, LiteLLMUsage)
    assert response.usage.get("total_tokens") == 37
    assert response.usage.prompt_tokens == 13
    assert response.usage.completion_tokens == 24


def test_combine_system_messages_none_string():
    existing = None
    new = "New"
    result = combine_system_messages(existing, new)
    assert result == "New"


def test_combine_system_messages_none_list():
    existing = None
    new = [{"type": "text", "text": "New"}]
    result = combine_system_messages(existing, new)
    assert result == [{"type": "text", "text": "New"}]


def test_combine_system_messages_invalid_type():
    with pytest.raises(ValueError):
        combine_system_messages(123, "New")


def test_extract_system_messages():
    messages = [
        {"role": "system", "content": "System message 1"},
        {"role": "user", "content": "User message"},
        {"role": "system", "content": "System message 2"},
    ]
    result = extract_system_messages(messages)
    expected = [
        {"type": "text", "text": "System message 1"},
        {"type": "text", "text": "System message 2"},
    ]
    assert result == expected


def test_extract_system_messages_no_system():
    messages = [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
    ]
    result = extract_system_messages(messages)
    assert result == []


def test_combine_system_messages_with_cache_control():
    existing = [
        {
            "type": "text",
            "text": "You are an AI assistant.",
        },
        {
            "type": "text",
            "text": "This is some context.",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    new = "Provide insightful analysis."
    result = combine_system_messages(existing, new)
    expected = [
        {
            "type": "text",
            "text": "You are an AI assistant.",
        },
        {
            "type": "text",
            "text": "This is some context.",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "Provide insightful analysis."},
    ]
    assert result == expected


def test_combine_system_messages_string_to_cache_control():
    existing = "You are an AI assistant."
    new = [
        {
            "type": "text",
            "text": "Analyze this text:",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "<long text content>"},
    ]
    result = combine_system_messages(existing, new)
    expected = [
        {"type": "text", "text": "You are an AI assistant."},
        {
            "type": "text",
            "text": "Analyze this text:",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "<long text content>"},
    ]
    assert result == expected


def test_extract_system_messages_with_cache_control():
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this text:",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": "User message"},
        {"role": "system", "content": "<long text content>"},
    ]
    result = extract_system_messages(messages)
    expected = [
        {"type": "text", "text": "You are an AI assistant."},
        {
            "type": "text",
            "text": "Analyze this text:",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "<long text content>"},
    ]
    assert result == expected


def test_combine_system_messages_preserve_cache_control():
    existing = [
        {
            "type": "text",
            "text": "You are an AI assistant.",
        },
        {
            "type": "text",
            "text": "This is some context.",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    new = [
        {
            "type": "text",
            "text": "Additional instruction.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    result = combine_system_messages(existing, new)
    expected = [
        {
            "type": "text",
            "text": "You are an AI assistant.",
        },
        {
            "type": "text",
            "text": "This is some context.",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": "Additional instruction.",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    assert result == expected


def test_provider_enum_covers_supported_providers():
    provider_values = {provider.value for provider in Provider}
    missing = [
        provider for provider in supported_providers if provider not in provider_values
    ]
    assert not missing, f"Missing providers in Provider enum: {missing}"


def test_get_provider_matches_supported_providers():
    provider_mapping = {
        "openai": Provider.OPENAI,
        "azure_openai": Provider.AZURE_OPENAI,
        "anyscale": Provider.ANYSCALE,
        "databricks": Provider.DATABRICKS,
        "anthropic": Provider.ANTHROPIC,
        "google": Provider.GOOGLE,
        "gemini": Provider.GEMINI,
        "generative-ai": Provider.GENERATIVE_AI,
        "vertexai": Provider.VERTEXAI,
        "mistral": Provider.MISTRAL,
        "cohere": Provider.COHERE,
        "perplexity": Provider.PERPLEXITY,
        "groq": Provider.GROQ,
        "writer": Provider.WRITER,
        "bedrock": Provider.BEDROCK,
        "cerebras": Provider.CEREBRAS,
        "deepseek": Provider.DEEPSEEK,
        "fireworks": Provider.FIREWORKS,
        "ollama": Provider.OLLAMA,
        "openrouter": Provider.OPENROUTER,
        "xai": Provider.XAI,
        "litellm": Provider.LITELLM,
        "together": Provider.TOGETHER,
        "atlascloud": Provider.ATLASCLOUD,
    }
    provider_urls = {
        "openai": "https://api.openai.com/v1",
        "azure_openai": "https://example.openai.azure.com",
        "anyscale": "https://api.endpoints.anyscale.com/v1",
        "databricks": "https://dbc-databricks.com",
        "anthropic": "https://api.anthropic.com",
        "google": "https://generativelanguage.googleapis.com",
        "gemini": "https://gemini.googleapis.com",
        "generative-ai": "https://generative-ai.googleapis.com",
        "vertexai": "https://vertexai.googleapis.com",
        "mistral": "https://api.mistral.ai",
        "cohere": "https://api.cohere.ai",
        "perplexity": "https://api.perplexity.ai",
        "groq": "https://api.groq.com",
        "writer": "https://api.writer.com",
        "bedrock": "https://bedrock.aws.amazon.com",
        "cerebras": "https://api.cerebras.ai",
        "deepseek": "https://api.deepseek.com",
        "fireworks": "https://api.fireworks.ai",
        "ollama": "http://localhost:11434",
        "openrouter": "https://openrouter.ai",
        "xai": "https://api.x.ai",
        "litellm": "https://litellm.ai",
        "together": "https://api.together.xyz/v1",
        "atlascloud": "https://api.atlascloud.ai/v1",
    }
    assert set(supported_providers) == set(provider_mapping)
    assert set(provider_mapping) == set(provider_urls)

    for provider_name, base_url in provider_urls.items():
        assert get_provider(base_url) == provider_mapping[provider_name]
