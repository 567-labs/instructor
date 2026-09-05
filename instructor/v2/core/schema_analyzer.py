"""Schema complexity pre-flight analyzer.

Analyzes Pydantic model JSON schemas before API calls to detect complexity
patterns that commonly cause LLM extraction failures. Provides actionable
warnings and mode recommendations based on schema characteristics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel


class Severity(Enum):
    """Severity level for schema analysis findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class SchemaFinding:
    """A single finding from schema analysis."""

    severity: Severity
    code: str
    message: str
    path: str = ""
    suggestion: str = ""


@dataclass
class SchemaAnalysis:
    """Complete analysis result for a schema."""

    findings: list[SchemaFinding] = field(default_factory=list)
    total_fields: int = 0
    max_depth: int = 0
    has_recursion: bool = False
    num_required: int = 0
    num_optional: int = 0
    estimated_token_overhead: int = 0

    @property
    def has_errors(self) -> bool:
        return any(f.severity == Severity.ERROR for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.severity == Severity.WARNING for f in self.findings)

    @property
    def complexity_score(self) -> int:
        """Score from 0-100 representing schema complexity.

        Higher scores indicate more complex schemas that are harder for LLMs.
        """
        score = 0
        score += min(self.total_fields * 2, 30)
        score += min(self.max_depth * 8, 32)
        score += 20 if self.has_recursion else 0
        score += min(self.estimated_token_overhead // 100, 18)
        return min(score, 100)

    @property
    def recommended_mode(self) -> str | None:
        """Suggest the best mode based on complexity characteristics."""
        if self.has_recursion:
            return "MD_JSON"
        if self.complexity_score > 70:
            return "JSON_SCHEMA"
        if self.max_depth > 4:
            return "TOOLS"
        return None


# Thresholds for warnings and errors
MAX_DEPTH_WARNING = 4
MAX_DEPTH_ERROR = 7
MAX_FIELDS_WARNING = 20
MAX_FIELDS_ERROR = 50
MAX_ENUM_VALUES_WARNING = 15
MAX_ENUM_VALUES_ERROR = 50
MAX_REQUIRED_FIELDS_WARNING = 12
MAX_PROPERTIES_PER_OBJECT_WARNING = 10
ESTIMATED_TOKENS_PER_FIELD = 15


def analyze_schema(
    response_model: type[BaseModel],
    *,
    warn: bool = False,
) -> SchemaAnalysis:
    """Analyze a Pydantic model's schema for complexity issues.

    Inspects the JSON schema generated from a Pydantic model and identifies
    patterns that commonly cause LLM extraction failures, including:

    - Deeply nested object hierarchies
    - Recursive/self-referencing models
    - Large enum value sets
    - Too many required fields
    - Excessively wide objects (many properties)

    Args:
        response_model: The Pydantic model class to analyze.
        warn: If True, emit Python warnings for findings at WARNING+ severity.

    Returns:
        SchemaAnalysis with findings and metrics.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Simple(BaseModel):
        ...     name: str
        ...     age: int
        >>> result = analyze_schema(Simple)
        >>> result.complexity_score < 20
        True

        >>> class Nested(BaseModel):
        ...     child: "Nested | None" = None
        >>> result = analyze_schema(Nested)
        >>> result.has_recursion
        True
    """
    schema = response_model.model_json_schema()
    analysis = SchemaAnalysis()

    defs = schema.get("$defs", {})

    _walk_schema(
        schema,
        analysis=analysis,
        defs=defs,
        path="$",
        depth=0,
        visited_refs=set(),
    )

    analysis.estimated_token_overhead = (
        analysis.total_fields * ESTIMATED_TOKENS_PER_FIELD
    )

    _check_global_thresholds(analysis)

    if warn:
        for finding in analysis.findings:
            if finding.severity in (Severity.WARNING, Severity.ERROR):
                msg = f"[instructor] {finding.code}: {finding.message}"
                if finding.suggestion:
                    msg += f" Suggestion: {finding.suggestion}"
                warnings.warn(msg, stacklevel=2)

    return analysis


def _walk_schema(
    node: dict[str, Any],
    *,
    analysis: SchemaAnalysis,
    defs: dict[str, Any],
    path: str,
    depth: int,
    visited_refs: set[str],
) -> None:
    """Recursively walk a JSON schema node and accumulate metrics."""
    analysis.max_depth = max(analysis.max_depth, depth)

    if "$ref" in node:
        ref_name = node["$ref"].split("/")[-1]
        if ref_name in visited_refs:
            analysis.has_recursion = True
            analysis.findings.append(
                SchemaFinding(
                    severity=Severity.WARNING,
                    code="RECURSIVE_REF",
                    message=f"Recursive reference detected: {ref_name}",
                    path=path,
                    suggestion=(
                        "Recursive schemas can confuse some LLMs. "
                        "Consider using MD_JSON mode or flattening the structure."
                    ),
                )
            )
            return
        if ref_name in defs:
            _walk_schema(
                defs[ref_name],
                analysis=analysis,
                defs=defs,
                path=f"{path}.${ref_name}",
                depth=depth,
                visited_refs=visited_refs | {ref_name},
            )
        return

    if "anyOf" in node or "oneOf" in node:
        variants = node.get("anyOf") or node.get("oneOf", [])
        for i, variant in enumerate(variants):
            _walk_schema(
                variant,
                analysis=analysis,
                defs=defs,
                path=f"{path}[variant_{i}]",
                depth=depth,
                visited_refs=visited_refs,
            )
        return

    if "allOf" in node:
        for i, sub in enumerate(node["allOf"]):
            _walk_schema(
                sub,
                analysis=analysis,
                defs=defs,
                path=f"{path}[allOf_{i}]",
                depth=depth,
                visited_refs=visited_refs,
            )
        return

    if "enum" in node:
        enum_values = node["enum"]
        count = len(enum_values)
        if count > MAX_ENUM_VALUES_ERROR:
            analysis.findings.append(
                SchemaFinding(
                    severity=Severity.ERROR,
                    code="ENUM_TOO_LARGE",
                    message=f"Enum at {path} has {count} values (max recommended: {MAX_ENUM_VALUES_ERROR})",
                    path=path,
                    suggestion="Split into categories or use a free-text field with validation.",
                )
            )
        elif count > MAX_ENUM_VALUES_WARNING:
            analysis.findings.append(
                SchemaFinding(
                    severity=Severity.WARNING,
                    code="ENUM_LARGE",
                    message=f"Enum at {path} has {count} values (>{MAX_ENUM_VALUES_WARNING})",
                    path=path,
                    suggestion="Large enums increase token usage and error rates.",
                )
            )
        return

    node_type = node.get("type")
    if node_type == "object" or "properties" in node:
        properties = node.get("properties", {})
        required = node.get("required", [])
        prop_count = len(properties)

        analysis.total_fields += prop_count
        analysis.num_required += len(required)
        analysis.num_optional += prop_count - len(required)

        if prop_count > MAX_PROPERTIES_PER_OBJECT_WARNING:
            analysis.findings.append(
                SchemaFinding(
                    severity=Severity.WARNING,
                    code="WIDE_OBJECT",
                    message=f"Object at {path} has {prop_count} properties",
                    path=path,
                    suggestion="Consider grouping related fields into sub-objects.",
                )
            )

        for prop_name, prop_schema in properties.items():
            _walk_schema(
                prop_schema,
                analysis=analysis,
                defs=defs,
                path=f"{path}.{prop_name}",
                depth=depth + 1,
                visited_refs=visited_refs,
            )
        return

    if node_type == "array":
        items = node.get("items", {})
        _walk_schema(
            items,
            analysis=analysis,
            defs=defs,
            path=f"{path}[]",
            depth=depth + 1,
            visited_refs=visited_refs,
        )
        return


def _check_global_thresholds(analysis: SchemaAnalysis) -> None:
    """Check global metrics against thresholds."""
    if analysis.max_depth > MAX_DEPTH_ERROR:
        analysis.findings.append(
            SchemaFinding(
                severity=Severity.ERROR,
                code="DEPTH_EXCESSIVE",
                message=f"Schema depth is {analysis.max_depth} (max recommended: {MAX_DEPTH_ERROR})",
                suggestion="Flatten nested structures or extract sub-models into separate calls.",
            )
        )
    elif analysis.max_depth > MAX_DEPTH_WARNING:
        analysis.findings.append(
            SchemaFinding(
                severity=Severity.WARNING,
                code="DEPTH_HIGH",
                message=f"Schema depth is {analysis.max_depth} (>{MAX_DEPTH_WARNING})",
                suggestion="Deep nesting increases extraction errors. Consider flattening.",
            )
        )

    if analysis.total_fields > MAX_FIELDS_ERROR:
        analysis.findings.append(
            SchemaFinding(
                severity=Severity.ERROR,
                code="TOO_MANY_FIELDS",
                message=f"Schema has {analysis.total_fields} total fields (max recommended: {MAX_FIELDS_ERROR})",
                suggestion="Split into multiple extraction calls or reduce schema scope.",
            )
        )
    elif analysis.total_fields > MAX_FIELDS_WARNING:
        analysis.findings.append(
            SchemaFinding(
                severity=Severity.WARNING,
                code="MANY_FIELDS",
                message=f"Schema has {analysis.total_fields} total fields (>{MAX_FIELDS_WARNING})",
                suggestion="Consider whether all fields are needed in a single extraction.",
            )
        )

    if analysis.num_required > MAX_REQUIRED_FIELDS_WARNING:
        analysis.findings.append(
            SchemaFinding(
                severity=Severity.WARNING,
                code="MANY_REQUIRED",
                message=f"Schema has {analysis.num_required} required fields (>{MAX_REQUIRED_FIELDS_WARNING})",
                suggestion="Make some fields optional with defaults to reduce extraction pressure.",
            )
        )
