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

        This heuristic combines structural depth, unique fields and recursion.
        """
        score = 0
        score += min(self.total_fields * 2, 30)
        score += min(self.max_depth * 8, 32)
        score += 20 if self.has_recursion else 0
        score += min(self.estimated_token_overhead // 100, 18)
        return min(score, 100)

    @property
    def recommended_mode(self) -> str | None:
        """Suggest a provider-independent starting point based on schema shape."""
        if self.has_recursion:
            return "MD_JSON"
        if self.complexity_score > 70:
            return "JSON_SCHEMA"
        if self.max_depth > 4:
            return "JSON_SCHEMA"
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

    analysis.max_depth = _schema_depth(schema, analysis)

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


def _schema_depth(schema: dict[str, Any], analysis: SchemaAnalysis) -> int:
    """Count declared fields once per definition and measure the deepest use."""
    definitions = schema.get("$defs", {})
    heights: dict[str, int] = {}
    active: set[str] = set()

    def walk(node: Any, path: str) -> int:
        if not isinstance(node, dict):
            return 0
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref[len("#/$defs/") :].replace("~1", "/").replace("~0", "~")
            if name in active:
                analysis.has_recursion = True
                analysis.findings.append(
                    SchemaFinding(
                        severity=Severity.WARNING,
                        code="RECURSIVE_REF",
                        message=f"Recursive reference detected: {name}",
                        path=path,
                        suggestion="Check whether the selected deployment accepts recursive schemas.",
                    )
                )
                return 0
            if name in definitions:
                if name not in heights:
                    active.add(name)
                    heights[name] = walk(definitions[name], f"{path}.${name}")
                    active.remove(name)
                return heights[name]
            return 0

        children = []
        for key in ("anyOf", "oneOf", "allOf"):
            children.extend(
                walk(variant, f"{path}.{key}[{index}]")
                for index, variant in enumerate(node.get(key, ()))
            )
        if "enum" in node:
            count = len(node["enum"])
            if count > MAX_ENUM_VALUES_WARNING:
                large = count > MAX_ENUM_VALUES_ERROR
                analysis.findings.append(
                    SchemaFinding(
                        severity=Severity.ERROR if large else Severity.WARNING,
                        code="ENUM_TOO_LARGE" if large else "ENUM_LARGE",
                        message=f"Enum at {path} has {count} values.",
                        path=path,
                        suggestion="Review the size of the enumerated choice set.",
                    )
                )
        properties = node.get("properties", {})
        if properties or node.get("type") == "object":
            required = set(node.get("required", ())).intersection(properties)
            count = len(properties)
            analysis.total_fields += count
            analysis.num_required += len(required)
            analysis.num_optional += count - len(required)
            if count > MAX_PROPERTIES_PER_OBJECT_WARNING:
                analysis.findings.append(
                    SchemaFinding(
                        severity=Severity.WARNING,
                        code="WIDE_OBJECT",
                        message=f"Object at {path} has {count} properties.",
                        path=path,
                        suggestion="Consider grouping related properties.",
                    )
                )
            children.extend(
                1 + walk(value, f"{path}.{key}") for key, value in properties.items()
            )
        if isinstance(node.get("additionalProperties"), dict):
            children.append(1 + walk(node["additionalProperties"], f"{path}.*"))
        if "items" in node:
            children.append(1 + walk(node["items"], f"{path}[]"))
        children.extend(
            1 + walk(value, f"{path}[{index}]")
            for index, value in enumerate(node.get("prefixItems", ()))
        )
        return max(children, default=0)

    return walk(schema, "$")


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
