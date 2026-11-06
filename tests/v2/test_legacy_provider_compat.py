from __future__ import annotations

import importlib


def test_legacy_provider_modules_remain_importable() -> None:
    modules = [
        "instructor.providers",
        "instructor.providers.anthropic.client",
        "instructor.providers.anthropic.utils",
        "instructor.providers.bedrock.client",
        "instructor.providers.bedrock.utils",
        "instructor.providers.cerebras.client",
        "instructor.providers.cerebras.utils",
        "instructor.providers.cohere.client",
        "instructor.providers.cohere.utils",
        "instructor.providers.fireworks.client",
        "instructor.providers.gemini.client",
        "instructor.providers.gemini.utils",
        "instructor.providers.genai.client",
        "instructor.providers.groq.client",
        "instructor.providers.mistral.client",
        "instructor.providers.mistral.utils",
        "instructor.providers.openai.utils",
        "instructor.providers.perplexity.client",
        "instructor.providers.perplexity.utils",
        "instructor.providers.vertexai.client",
        "instructor.providers.writer.client",
        "instructor.providers.writer.utils",
        "instructor.providers.xai.client",
        "instructor.providers.xai.utils",
    ]

    for module_name in modules:
        importlib.import_module(module_name)


def test_legacy_provider_utils_forward_to_v2_symbols() -> None:
    openai_utils = importlib.import_module("instructor.providers.openai.utils")
    anthropic_utils = importlib.import_module("instructor.providers.anthropic.utils")
    gemini_utils = importlib.import_module("instructor.providers.gemini.utils")

    assert (
        openai_utils.reask_tools
        is importlib.import_module(
            "instructor.v2.providers.openai.handlers"
        ).reask_tools
    )
    assert (
        anthropic_utils.combine_system_messages
        is importlib.import_module(
            "instructor.v2.providers.anthropic.handlers"
        ).combine_system_messages
    )
    assert (
        gemini_utils.map_to_gemini_function_schema
        is importlib.import_module(
            "instructor.v2.providers.gemini.utils"
        ).map_to_gemini_function_schema
    )
