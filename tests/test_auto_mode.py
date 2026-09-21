"""Tests for the advisory automatic mode selection system.

Covers simple models, recursive schemas, deep nesting, high complexity,
provider-specific constraints, and edge cases around fallback behavior.
"""

from __future__ import annotations

from pydantic import BaseModel

from instructor.v2.core.auto_mode import select_mode
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.schema_analyzer import analyze_schema


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


# High complexity chain (leaf to root)
class ComplexBottomA(BaseModel):
    z1: str
    z2: str
    z3: str
    z4: str
    z5: str
    z6: str
    z7: str


class ComplexDeeperA(BaseModel):
    bottom: ComplexBottomA
    y1: str
    y2: str
    y3: str


class ComplexDeepA(BaseModel):
    deeper: ComplexDeeperA
    x1: str
    x2: str
    x3: str
    x4: str
    x5: str


class ComplexInnerA(BaseModel):
    deep: ComplexDeepA
    w1: str
    w2: str
    w3: str
    w4: str
    w5: str


class ComplexSectionA(BaseModel):
    inner: ComplexInnerA
    v1: str
    v2: str
    v3: str
    v4: str
    v5: str


class ComplexInnerB(BaseModel):
    n1: str
    n2: str
    n3: str
    n4: str
    n5: str
    n6: str
    n7: str
    n8: str
    n9: str
    n10: str
    n11: str
    n12: str


class ComplexSectionB(BaseModel):
    inner: ComplexInnerB
    m1: str
    m2: str
    m3: str
    m4: str
    m5: str
    m6: str
    m7: str


class HighComplexityModel(BaseModel):
    """A model designed to exceed the 70-point complexity threshold.

    Combines deep nesting (5+ levels) with many fields to push the score over 70.
    """

    section_a: ComplexSectionA
    section_b: ComplexSectionB
    f01: str
    f02: str
    f03: str
    f04: str
    f05: str
    f06: str
    f07: str
    f08: str
    f09: str
    f10: str


# Medium complexity chain (leaf to root)
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

    def test_deep_bedrock_falls_to_tools(self):
        # Bedrock supports TOOLS and MD_JSON only
        mode = select_mode(DeeplyNestedModel, Provider.BEDROCK)
        assert mode in (Mode.TOOLS, Mode.MD_JSON)


class TestHighComplexitySelection:
    def test_high_complexity_strict_prefers_json_schema(self):
        analysis = analyze_schema(HighComplexityModel)
        assert analysis.complexity_score > 70, (
            f"Test model score {analysis.complexity_score} too low, expected >70"
        )
        mode = select_mode(HighComplexityModel, Provider.OPENAI, prefer_strict=True)
        assert mode == Mode.JSON_SCHEMA

    def test_high_complexity_relaxed_prefers_md_json(self):
        mode = select_mode(HighComplexityModel, Provider.OPENAI, prefer_strict=False)
        assert mode == Mode.MD_JSON

    def test_high_complexity_mistral(self):
        mode = select_mode(HighComplexityModel, Provider.MISTRAL, prefer_strict=True)
        assert mode == Mode.JSON_SCHEMA


class TestProviderConstraints:
    def test_perplexity_only_supports_md_json(self):
        mode = select_mode(SimpleUser, Provider.PERPLEXITY)
        assert mode == Mode.MD_JSON

    def test_bedrock_no_json_schema(self):
        mode = select_mode(HighComplexityModel, Provider.BEDROCK, prefer_strict=True)
        assert mode in (Mode.TOOLS, Mode.MD_JSON)

    def test_genai_supports_tools_and_json(self):
        mode = select_mode(SimpleUser, Provider.GENAI)
        assert mode in (Mode.TOOLS, Mode.JSON)


class TestPrecomputedAnalysis:
    def test_accepts_precomputed_analysis(self):
        analysis = analyze_schema(SimpleUser)
        mode = select_mode(SimpleUser, Provider.OPENAI, analysis=analysis)
        assert mode == Mode.TOOLS

    def test_precomputed_recursive_analysis(self):
        analysis = analyze_schema(RecursiveTree)
        mode = select_mode(RecursiveTree, Provider.OPENAI, analysis=analysis)
        assert mode == Mode.MD_JSON


class TestFallbackBehavior:
    def test_unknown_provider_defaults_to_tools(self):
        mode = select_mode(SimpleUser, Provider.UNKNOWN)
        assert mode == Mode.TOOLS

    def test_ollama_defaults_to_tools(self):
        mode = select_mode(SimpleUser, Provider.OLLAMA)
        assert mode == Mode.TOOLS


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


class TestIntegrationWithAnalyzer:
    def test_analysis_and_mode_selection_consistent(self):
        analysis = analyze_schema(RecursiveTree)
        assert analysis.has_recursion
        mode = select_mode(RecursiveTree, Provider.OPENAI, analysis=analysis)
        assert mode == Mode.MD_JSON

    def test_simple_analysis_and_mode_consistent(self):
        analysis = analyze_schema(SimpleUser)
        assert not analysis.has_recursion
        assert analysis.complexity_score < 30
        mode = select_mode(SimpleUser, Provider.OPENAI, analysis=analysis)
        assert mode == Mode.TOOLS
