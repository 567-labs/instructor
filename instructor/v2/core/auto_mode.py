"""Opt-in mode suggestions based on schema shape and declared provider support."""

from __future__ import annotations

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.provider_specs import PROVIDER_SPECS
from instructor.v2.core.providers import Provider
from instructor.v2.core.schema_analyzer import SchemaAnalysis, analyze_schema


def select_mode(
    response_model: type[BaseModel],
    provider: Provider,
    *,
    prefer_strict: bool = True,
    analysis: SchemaAnalysis | None = None,
) -> Mode:
    """Suggest a declared mode for a model without changing client defaults.

    The ordering is a heuristic. A specific model or deployment may support
    fewer modes than its provider. For providers without declared modes,
    choose a mode supported by the deployment or use its client default.
    """
    spec = PROVIDER_SPECS.get(provider)
    if spec is None or not spec.supported_modes:
        raise ValueError(
            f"No structured-output modes are declared for {provider.value}"
        )

    if analysis is None:
        analysis = analyze_schema(response_model)
    if analysis.has_recursion:
        preference = (Mode.MD_JSON, Mode.JSON, Mode.JSON_SCHEMA, Mode.TOOLS)
    elif prefer_strict and (analysis.complexity_score > 70 or analysis.max_depth > 4):
        preference = (Mode.JSON_SCHEMA, Mode.TOOLS, Mode.MD_JSON, Mode.JSON)
    elif not prefer_strict and analysis.complexity_score > 70:
        preference = (Mode.MD_JSON, Mode.JSON_SCHEMA, Mode.TOOLS, Mode.JSON)
    elif prefer_strict or analysis.max_depth > 4:
        preference = (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.JSON)
    else:
        preference = (Mode.TOOLS, Mode.MD_JSON, Mode.JSON_SCHEMA, Mode.JSON)
    return next(
        (mode for mode in preference if mode in spec.supported_modes),
        spec.supported_modes[0],
    )
