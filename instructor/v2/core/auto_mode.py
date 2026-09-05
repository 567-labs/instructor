"""Advisory automatic mode selection based on provider capabilities.

Selects modes using schema complexity and provider support.

This module is strictly advisory. It provides a recommendation that users can
pass to from_openai(..., mode=recommended) or ignore entirely. It does NOT
automatically override user-specified modes, and it is NOT called internally
by from_provider or any other client factory.

The intent is to give users a starting point when they are unsure which mode
to use for a given schema, especially when dealing with complex or recursive
models that behave differently across providers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.provider_specs import PROVIDER_SPECS
from instructor.v2.core.schema_analyzer import (
    SchemaAnalysis,
    analyze_schema,
)

if TYPE_CHECKING:
    from instructor.v2.core.provider_specs import ProviderSpec

logger = logging.getLogger("instructor.v2.auto_mode")


def select_mode(
    response_model: type[BaseModel],
    provider: Provider,
    *,
    prefer_strict: bool = True,
    analysis: SchemaAnalysis | None = None,
) -> Mode:
    """Return an advisory mode recommendation for a provider and response model.

    This function is purely advisory. It does NOT automatically override any
    user-specified mode, and it is NOT called internally by from_provider or
    any client factory. Users opt in explicitly:

        mode = select_mode(MyModel, Provider.OPENAI)
        client = instructor.from_openai(openai.OpenAI(), mode=mode)

    The recommendation is based on:
    - Provider support (not all providers support all modes)
    - Schema complexity (simple schemas work with TOOLS, complex with JSON_SCHEMA)
    - Recursion (recursive models need MD_JSON or JSON modes)
    - Depth (deeply nested schemas benefit from JSON_SCHEMA)

    Args:
        response_model: The Pydantic model to extract.
        provider: The target LLM provider.
        prefer_strict: If True, prefer stricter modes (JSON_SCHEMA over MD_JSON)
            when both are available and suitable. Default True.
        analysis: Pre-computed schema analysis. If None, computed internally.

    Returns:
        The recommended Mode enum value. Users should pass this to their
        client factory or ignore it if they have a reason to prefer a
        specific mode.

    Note:
        This is advisory only and not integrated into from_provider.
        The heuristics are intentionally conservative - when in doubt,
        TOOLS mode is preferred since providers optimize for it.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Simple(BaseModel):
        ...     name: str
        >>> select_mode(Simple, Provider.OPENAI)
        <Mode.TOOLS: 'tool_call'>

        >>> class Complex(BaseModel):
        ...     items: list["Complex"]
        >>> select_mode(Complex, Provider.OPENAI)
        <Mode.MD_JSON: 'markdown_json_mode'>
    """
    if analysis is None:
        analysis = analyze_schema(response_model)

    spec = PROVIDER_SPECS.get(provider)
    supported = _get_supported_modes(spec)

    if not supported:
        logger.debug(f"No supported modes found for {provider}, defaulting to TOOLS")
        return Mode.TOOLS

    if analysis.has_recursion:
        return _pick_recursive_mode(supported, prefer_strict)

    if analysis.complexity_score > 70:
        return _pick_high_complexity_mode(supported, prefer_strict)

    if analysis.max_depth > 4:
        return _pick_deep_schema_mode(supported, prefer_strict)

    return _pick_simple_mode(supported, prefer_strict)


def _get_supported_modes(spec: ProviderSpec | None) -> set[Mode]:
    """Extract supported modes from a provider spec."""
    if spec is None:
        return {Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON}
    return set(spec.supported_modes)


def _pick_recursive_mode(supported: set[Mode], prefer_strict: bool) -> Mode:  # noqa: ARG001
    """Pick mode for recursive schemas.

    Recursive models cannot be represented in tool-call schemas on most providers.
    MD_JSON or JSON modes handle them gracefully since the LLM generates free-form
    JSON that is then validated.
    """
    preference_order = [Mode.MD_JSON, Mode.JSON, Mode.JSON_SCHEMA, Mode.TOOLS]
    return _first_supported(preference_order, supported)


def _pick_high_complexity_mode(supported: set[Mode], prefer_strict: bool) -> Mode:
    """Pick mode for highly complex schemas (score > 70).

    JSON_SCHEMA mode gives the LLM explicit structural guidance, reducing errors
    on complex schemas. Falls back to MD_JSON, then TOOLS.
    """
    if prefer_strict:
        preference_order = [Mode.JSON_SCHEMA, Mode.TOOLS, Mode.MD_JSON, Mode.JSON]
    else:
        preference_order = [Mode.MD_JSON, Mode.JSON_SCHEMA, Mode.TOOLS, Mode.JSON]
    return _first_supported(preference_order, supported)


def _pick_deep_schema_mode(supported: set[Mode], prefer_strict: bool) -> Mode:
    """Pick mode for deeply nested schemas (depth > 4).

    TOOLS mode can struggle with deep nesting since each level adds function
    call overhead. JSON_SCHEMA provides the full structure inline.
    """
    if prefer_strict:
        preference_order = [Mode.JSON_SCHEMA, Mode.TOOLS, Mode.MD_JSON, Mode.JSON]
    else:
        preference_order = [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.JSON]
    return _first_supported(preference_order, supported)


def _pick_simple_mode(supported: set[Mode], prefer_strict: bool) -> Mode:
    """Pick mode for simple schemas (low complexity, shallow depth).

    TOOLS mode is most reliable for simple schemas since it leverages native
    function calling which providers optimize for.
    """
    if prefer_strict:
        preference_order = [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.JSON]
    else:
        preference_order = [Mode.TOOLS, Mode.MD_JSON, Mode.JSON_SCHEMA, Mode.JSON]
    return _first_supported(preference_order, supported)


def _first_supported(preference: list[Mode], supported: set[Mode]) -> Mode:
    """Return the first mode from preference that is in the supported set."""
    for mode in preference:
        if mode in supported:
            return mode
    return Mode.TOOLS
