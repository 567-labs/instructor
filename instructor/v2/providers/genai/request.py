"""Request configuration policy for the Google GenAI provider."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

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


def apply_openai_generation_config(
    generation_config: dict[str, Any], base_config: dict[str, Any]
) -> dict[str, Any]:
    """Move OpenAI-style generation options into GenAI config keys."""
    for openai_key, gemini_key in OPENAI_TO_GEMINI_MAP.items():
        if openai_key in generation_config:
            val = generation_config.pop(openai_key)
            if val is not None:
                base_config[gemini_key] = val
    return base_config


def _extract_user_config_fields(user_config: Any) -> dict[str, Any]:
    """Extract non-None fields from a user config dict, model, or object."""
    if user_config is None:
        return {}
    if isinstance(user_config, dict):
        items = user_config.items()
    elif isinstance(user_config, BaseModel):
        items = (
            (field, getattr(user_config, field))
            for field in type(user_config).model_fields
        )
    else:
        items = vars(user_config).items()
    return {key: value for key, value in items if value is not None}


def update_genai_kwargs(
    kwargs: dict[str, Any], base_config: dict[str, Any]
) -> dict[str, Any]:
    """Merge GenAI request options into the provider configuration."""
    from google.genai.types import HarmBlockThreshold, HarmCategory

    new_kwargs = kwargs.copy()
    owned_keys = {
        k
        for k, v in base_config.items()
        if v is not None and not (k == "system_instruction" and v == "")
    }

    raw_generation_config = new_kwargs.pop("generation_config", None)
    generation_config = (
        dict(raw_generation_config) if isinstance(raw_generation_config, dict) else {}
    )
    apply_openai_generation_config(generation_config, base_config)

    user_fields = _extract_user_config_fields(new_kwargs.get("config"))

    safety_settings = new_kwargs.pop("safety_settings", None)
    if safety_settings is None:
        safety_settings = user_fields.get("safety_settings")
    if safety_settings is None:
        safety_settings = {}
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

    thinking_config = new_kwargs.pop("thinking_config", None)
    if thinking_config is None:
        thinking_config = user_fields.get("thinking_config")

    if thinking_config is not None:
        base_config["thinking_config"] = thinking_config

    for field, field_value in user_fields.items():
        if field in ("safety_settings", "thinking_config"):
            continue
        if field not in owned_keys:
            base_config[field] = field_value

    cached_content = new_kwargs.pop("cached_content", None)
    if cached_content is not None and "cached_content" not in base_config:
        base_config["cached_content"] = cached_content
    if base_config.get("cached_content") is not None:
        # Cached resources own these fields; sending them again is invalid.
        for field in ("system_instruction", "tools", "tool_config"):
            base_config.pop(field, None)

    return base_config
