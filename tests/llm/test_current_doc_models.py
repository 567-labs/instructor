"""Exercise the public model IDs used in the current provider guides."""

import os
from typing import Any

import pytest
from pydantic import BaseModel

import instructor


class ExtractedNumber(BaseModel):
    value: int


@pytest.mark.llm
@pytest.mark.parametrize(
    "model,credential",
    [
        ("openai/gpt-5.6-luna", "OPENAI_API_KEY"),
        ("anthropic/claude-sonnet-5", "ANTHROPIC_API_KEY"),
        ("google/gemini-3.8-flash", "GOOGLE_API_KEY"),
        ("cohere/command-a-03-2025", "COHERE_API_KEY"),
        ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY"),
        ("mistral/mistral-small-latest", "MISTRAL_API_KEY"),
        ("fireworks/accounts/fireworks/models/kimi-k2p5", "FIREWORKS_API_KEY"),
        ("cerebras/gpt-oss-120b", "CEREBRAS_API_KEY"),
        ("writer/palmyra-x5", "WRITER_API_KEY"),
        ("xai/grok-4.20-reasoning", "XAI_API_KEY"),
        ("perplexity/sonar", "PERPLEXITY_API_KEY"),
        ("deepseek/deepseek-v4-flash", "DEEPSEEK_API_KEY"),
        ("openrouter/google/gemini-3.8-flash", "OPENROUTER_API_KEY"),
        ("together/meta-llama/Llama-3.3-70B-Instruct-Turbo", "TOGETHER_API_KEY"),
    ],
)
def test_current_documented_model(model: str, credential: str) -> None:
    if not os.getenv(credential):
        pytest.skip(f"{credential} is not configured")
    mode = (
        instructor.Mode.MD_JSON
        if model.startswith("perplexity/")
        else instructor.Mode.TOOLS
    )
    client = instructor.from_provider(model, mode=mode)
    options: dict[str, Any] = (
        {"max_completion_tokens": 2048, "reasoning_effort": "none"}
        if model.startswith("openai/")
        else {"max_tokens": 2048}
    )
    result = client.create(
        response_model=ExtractedNumber,
        messages=[{"role": "user", "content": "Extract the integer 42."}],
        max_retries=1,
        **options,
    )
    assert result.value == 42
