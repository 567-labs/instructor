"""Tests for the advisory automatic mode selection system.

Covers simple models, recursive schemas, deep nesting, high complexity,
provider-specific constraints, and edge cases around fallback behavior.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from instructor.v2.core.auto_mode import select_mode
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.schema_analyzer import SchemaAnalysis, analyze_schema


# --- Test Models (ordered so dependencies come before usage) ---


class SimpleUser(BaseModel):
    name: str
    age: int


class RecursiveTree(BaseModel):
    label: str
    children: list[RecursiveTree] = []


class OptionalRecursive(BaseModel):
    name: str
    parent: OptionalRecursive | None = None


# Deep nesting chain (leaf to root)
class LevelE(BaseModel):
    value: str


class LevelD(BaseModel):
    e: LevelE


class LevelC(BaseModel):
    d: LevelD


class LevelB(BaseModel):
    c: LevelC


class LevelA(BaseModel):
    b: LevelB


class DeeplyNestedModel(BaseModel):
    a: LevelA


class RefItem(BaseModel):
    url: str
    label: str


class MetaBlock(BaseModel):
    author: str
    version: int
    refs: list[RefItem]


class MediumComplexity(BaseModel):
    title: str
    description: str
    tags: list[str]
    metadata: MetaBlock


# --- Tests ---


class TestSimpleModelSelection:
    def test_openai_simple_picks_tools(self):
        mode = select_mode(SimpleUser, Provider.OPENAI)
        assert mode == Mode.TOOLS

    def test_anthropic_simple_picks_tools(self):
        mode = select_mode(SimpleUser, Provider.ANTHROPIC)
        assert mode == Mode.TOOLS

    def test_genai_simple_picks_tools(self):
        mode = select_mode(SimpleUser, Provider.GENAI)
        assert mode == Mode.TOOLS

    def test_cohere_simple_picks_tools(self):
        mode = select_mode(SimpleUser, Provider.COHERE)
        assert mode == Mode.TOOLS

    def test_mistral_simple_picks_tools(self):
        mode = select_mode(SimpleUser, Provider.MISTRAL)
        assert mode == Mode.TOOLS


class TestRecursiveModelSelection:
    def test_recursive_prefers_md_json_openai(self):
        mode = select_mode(RecursiveTree, Provider.OPENAI)
        assert mode == Mode.MD_JSON

    def test_recursive_prefers_md_json_anthropic(self):
        mode = select_mode(RecursiveTree, Provider.ANTHROPIC)
        # Anthropic does not support MD_JSON, should fallback to JSON
        assert mode in (Mode.MD_JSON, Mode.JSON)

    def test_recursive_prefers_md_json_genai(self):
        mode = select_mode(RecursiveTree, Provider.GENAI)
        # GenAI supports JSON but not MD_JSON
        assert mode == Mode.JSON

    def test_optional_recursive_also_detected(self):
        mode = select_mode(OptionalRecursive, Provider.OPENAI)
        assert mode == Mode.MD_JSON


class TestDeepNestingSelection:
    def test_deep_openai_prefers_json_schema_strict(self):
        mode = select_mode(DeeplyNestedModel, Provider.OPENAI, prefer_strict=True)
        assert mode == Mode.JSON_SCHEMA

    def test_deep_openai_prefers_tools_relaxed(self):
        mode = select_mode(DeeplyNestedModel, Provider.OPENAI, prefer_strict=False)
        assert mode == Mode.TOOLS

    def test_deep_bedrock_uses_declared_json_schema(self):
        mode = select_mode(DeeplyNestedModel, Provider.BEDROCK)
        assert mode is Mode.JSON_SCHEMA


@pytest.mark.parametrize(
    ("overhead", "score", "provider", "strict", "expected"),
    [
        (1600, 70, Provider.OPENAI, True, Mode.TOOLS),
        (1700, 71, Provider.OPENAI, True, Mode.JSON_SCHEMA),
        (1700, 71, Provider.OPENAI, False, Mode.MD_JSON),
        (1700, 71, Provider.MISTRAL, True, Mode.JSON_SCHEMA),
        (1700, 71, Provider.BEDROCK, True, Mode.JSON_SCHEMA),
    ],
)
def test_shallow_precomputed_score_controls_mode(
    overhead: int, score: int, provider: Provider, strict: bool, expected: Mode
) -> None:
    analysis = SchemaAnalysis(
        total_fields=15, max_depth=3, estimated_token_overhead=overhead
    )
    assert analysis.complexity_score == score
    assert (
        select_mode(SimpleUser, provider, analysis=analysis, prefer_strict=strict)
        is expected
    )


class TestProviderConstraints:
    def test_perplexity_only_supports_md_json(self):
        mode = select_mode(SimpleUser, Provider.PERPLEXITY)
        assert mode == Mode.MD_JSON

    def test_genai_supports_tools_and_json(self):
        mode = select_mode(SimpleUser, Provider.GENAI)
        assert mode in (Mode.TOOLS, Mode.JSON)


def test_precomputed_recursion_overrides_the_model_default() -> None:
    analysis = analyze_schema(RecursiveTree)
    assert select_mode(SimpleUser, Provider.OPENAI, analysis=analysis) is Mode.MD_JSON


class TestFallbackBehavior:
    @pytest.mark.parametrize("provider", [Provider.UNKNOWN, Provider.OLLAMA])
    def test_no_declared_modes_requires_deployment_choice(self, provider):
        with pytest.raises(ValueError, match="No structured-output modes"):
            select_mode(SimpleUser, provider)

    def test_fallback_respects_declared_modes(self, monkeypatch):
        from dataclasses import replace
        from instructor.v2.core import auto_mode

        spec = auto_mode.PROVIDER_SPECS[Provider.OPENAI]
        monkeypatch.setattr(
            auto_mode,
            "PROVIDER_SPECS",
            {
                Provider.OPENAI: replace(
                    spec, supported_modes=(Mode.RESPONSES_TOOLS, Mode.PARALLEL_TOOLS)
                ),
            },
        )
        assert select_mode(SimpleUser, Provider.OPENAI) is Mode.RESPONSES_TOOLS


class TestPreferStrictFlag:
    def test_prefer_strict_true_vs_false_simple(self):
        strict = select_mode(SimpleUser, Provider.OPENAI, prefer_strict=True)
        relaxed = select_mode(SimpleUser, Provider.OPENAI, prefer_strict=False)
        assert strict == relaxed == Mode.TOOLS

    def test_prefer_strict_affects_deep_model(self):
        strict = select_mode(DeeplyNestedModel, Provider.OPENAI, prefer_strict=True)
        relaxed = select_mode(DeeplyNestedModel, Provider.OPENAI, prefer_strict=False)
        assert strict == Mode.JSON_SCHEMA
        assert relaxed == Mode.TOOLS


class TestMediumComplexity:
    def test_medium_model_picks_tools(self):
        result = analyze_schema(MediumComplexity)
        assert result.complexity_score < 70
        mode = select_mode(MediumComplexity, Provider.OPENAI)
        assert mode == Mode.TOOLS
