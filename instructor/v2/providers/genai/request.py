"""Request configuration policy for the Google GenAI provider."""

from __future__ import annotations

from typing import Any


def update_genai_kwargs(
    kwargs: dict[str, Any], base_config: dict[str, Any]
) -> dict[str, Any]:
    """Merge GenAI request options into the provider configuration."""
    from google.genai.types import HarmBlockThreshold, HarmCategory

    new_kwargs = kwargs.copy()

    OPENAI_TO_GEMINI_MAP = {
        "max_tokens": "max_output_tokens",
        "temperature": "temperature",
        "n": "candidate_count",
        "top_p": "top_p",
        "stop": "stop_sequences",
        "seed": "seed",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
    }

    generation_config = new_kwargs.pop("generation_config", {})

    for openai_key, gemini_key in OPENAI_TO_GEMINI_MAP.items():
        if openai_key in generation_config:
            val = generation_config.pop(openai_key)
            if val is not None:
                base_config[gemini_key] = val

    safety_settings = new_kwargs.pop("safety_settings", {})
    base_config["safety_settings"] = []

    if isinstance(safety_settings, list):
        base_config["safety_settings"] = safety_settings
        safety_settings = None

    excluded_categories = {HarmCategory.HARM_CATEGORY_UNSPECIFIED}
    if hasattr(HarmCategory, "HARM_CATEGORY_JAILBREAK"):
        excluded_categories.add(HarmCategory.HARM_CATEGORY_JAILBREAK)

    if safety_settings is not None:
        text_categories = [
            c
            for c in HarmCategory
            if c not in excluded_categories
            and not c.name.startswith("HARM_CATEGORY_IMAGE_")
        ]

        for category in text_categories:
            threshold = HarmBlockThreshold.OFF
            if isinstance(safety_settings, dict):
                if category in safety_settings:
                    threshold = safety_settings[category]

            base_config["safety_settings"].append(
                {
                    "category": category,
                    "threshold": threshold,
                }
            )

    user_config = new_kwargs.get("config")
    user_thinking_config = None
    if isinstance(user_config, dict):
        user_thinking_config = user_config.get("thinking_config")
    elif user_config is not None and hasattr(user_config, "thinking_config"):
        user_thinking_config = user_config.thinking_config

    thinking_config = new_kwargs.pop("thinking_config", None)
    if thinking_config is None:
        thinking_config = user_thinking_config

    if thinking_config is not None:
        base_config["thinking_config"] = thinking_config

    if user_config is not None:
        config_fields_to_merge = [
            "automatic_function_calling",
            "labels",
            "cached_content",
        ]
        for field in config_fields_to_merge:
            if isinstance(user_config, dict):
                field_value = user_config.get(field)
            elif hasattr(user_config, field):
                field_value = getattr(user_config, field)
            else:
                field_value = None

            if field_value is not None and field not in base_config:
                base_config[field] = field_value

    cached_content = new_kwargs.pop("cached_content", None)
    if cached_content is not None and "cached_content" not in base_config:
        base_config["cached_content"] = cached_content
    if base_config.get("cached_content") is not None:
        # Cached resources own these fields; sending them again is invalid.
        for field in ("system_instruction", "tools", "tool_config"):
            base_config.pop(field, None)

    return base_config
