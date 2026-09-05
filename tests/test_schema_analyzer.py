"""Tests for the schema complexity pre-flight analyzer.

Covers simple schemas, deeply nested models, recursive structures, large enums,
wide objects, and mixed complexity patterns to ensure accurate detection and
scoring.
"""

from __future__ import annotations

import warnings
from enum import Enum

from pydantic import BaseModel

from instructor.v2.core.schema_analyzer import (
    SchemaFinding,
    Severity,
    analyze_schema,
)


# --- Test Models (ordered so dependencies come before usage) ---


class SimpleModel(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


class NestedModel(BaseModel):
    address: Address
    name: str


class RecursiveModel(BaseModel):
    value: str
    children: list[RecursiveModel] = []


class OptionalRecursiveModel(BaseModel):
    name: str
    parent: OptionalRecursiveModel | None = None


class Level5(BaseModel):
    value: str


class Level4(BaseModel):
    level5: Level5


class Level3(BaseModel):
    level4: Level4


class Level2(BaseModel):
    level3: Level3


class Level1(BaseModel):
    level2: Level2


class DeeplyNested(BaseModel):
    level1: Level1


class WideModel(BaseModel):
    field_01: str
    field_02: str
    field_03: str
    field_04: str
    field_05: str
    field_06: str
    field_07: str
    field_08: str
    field_09: str
    field_10: str
    field_11: str
    field_12: str


BigValues = [f"cat_{i}" for i in range(25)]
BigCategory = Enum("BigCategory", {v: v for v in BigValues})


class ModelWithLargeEnum(BaseModel):
    category: BigCategory


HugeValues = [f"item_{i}" for i in range(60)]
HugeEnum = Enum("HugeEnum", {v: v for v in HugeValues})


class ModelWithHugeEnum(BaseModel):
    item: HugeEnum


class ModelWithManyRequired(BaseModel):
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
    f11: str
    f12: str
    f13: str


class EmptyModel(BaseModel):
    pass


class DataA(BaseModel):
    kind: str = "a"
    value_a: int


class DataB(BaseModel):
    kind: str = "b"
    value_b: str


class ModelWithUnion(BaseModel):
    data: DataA | DataB


class ModelWithOptionals(BaseModel):
    required_field: str
    optional_1: str | None = None
    optional_2: int | None = None
    optional_3: float | None = None


class ListItem(BaseModel):
    name: str
    quantity: int


class ModelWithList(BaseModel):
    items: list[ListItem]


# --- Tests ---


class TestSimpleSchemas:
    def test_simple_model_low_complexity(self):
        result = analyze_schema(SimpleModel)
        assert result.complexity_score < 20
        assert result.total_fields == 2
        assert result.max_depth == 1
        assert not result.has_recursion
        assert not result.has_errors
        assert not result.has_warnings

    def test_empty_model(self):
        result = analyze_schema(EmptyModel)
        assert result.complexity_score == 0
        assert result.total_fields == 0
        assert result.max_depth == 0
        assert not result.has_recursion

    def test_model_with_optionals(self):
        result = analyze_schema(ModelWithOptionals)
        assert result.num_required == 1
        assert result.num_optional == 3
        assert result.total_fields == 4

    def test_nested_model_adds_depth(self):
        result = analyze_schema(NestedModel)
        assert result.max_depth >= 2
        assert result.total_fields >= 5


class TestRecursiveSchemas:
    def test_recursive_model_detected(self):
        result = analyze_schema(RecursiveModel)
        assert result.has_recursion
        assert any(f.code == "RECURSIVE_REF" for f in result.findings)

    def test_optional_recursive_model_detected(self):
        result = analyze_schema(OptionalRecursiveModel)
        assert result.has_recursion

    def test_recursive_model_recommends_md_json(self):
        result = analyze_schema(RecursiveModel)
        assert result.recommended_mode == "MD_JSON"


class TestDeepNesting:
    def test_deeply_nested_reports_high_depth(self):
        result = analyze_schema(DeeplyNested)
        assert result.max_depth >= 5

    def test_depth_warning_threshold(self):
        result = analyze_schema(DeeplyNested)
        depth_findings = [
            f for f in result.findings if f.code in ("DEPTH_HIGH", "DEPTH_EXCESSIVE")
        ]
        assert len(depth_findings) > 0


class TestWideObjects:
    def test_wide_model_triggers_warning(self):
        result = analyze_schema(WideModel)
        wide_findings = [f for f in result.findings if f.code == "WIDE_OBJECT"]
        assert len(wide_findings) > 0

    def test_wide_model_counts_fields_correctly(self):
        result = analyze_schema(WideModel)
        assert result.total_fields == 12


class TestEnums:
    def test_large_enum_triggers_warning(self):
        result = analyze_schema(ModelWithLargeEnum)
        enum_findings = [f for f in result.findings if f.code == "ENUM_LARGE"]
        assert len(enum_findings) > 0

    def test_huge_enum_triggers_error(self):
        result = analyze_schema(ModelWithHugeEnum)
        enum_findings = [f for f in result.findings if f.code == "ENUM_TOO_LARGE"]
        assert len(enum_findings) > 0


class TestRequiredFields:
    def test_many_required_triggers_warning(self):
        result = analyze_schema(ModelWithManyRequired)
        required_findings = [f for f in result.findings if f.code == "MANY_REQUIRED"]
        assert len(required_findings) > 0

    def test_required_count_accurate(self):
        result = analyze_schema(ModelWithManyRequired)
        assert result.num_required == 13


class TestUnionTypes:
    def test_union_model_handles_variants(self):
        result = analyze_schema(ModelWithUnion)
        assert result.total_fields >= 2
        assert not result.has_recursion

    def test_union_model_low_complexity(self):
        result = analyze_schema(ModelWithUnion)
        assert result.complexity_score < 50


class TestListFields:
    def test_list_model_adds_depth(self):
        result = analyze_schema(ModelWithList)
        assert result.max_depth >= 2

    def test_list_item_fields_counted(self):
        result = analyze_schema(ModelWithList)
        assert result.total_fields >= 3


class TestComplexityScore:
    def test_score_bounded_0_to_100(self):
        for model in [SimpleModel, RecursiveModel, DeeplyNested, WideModel]:
            result = analyze_schema(model)
            assert 0 <= result.complexity_score <= 100

    def test_simple_lower_than_recursive(self):
        simple = analyze_schema(SimpleModel)
        recursive = analyze_schema(RecursiveModel)
        assert simple.complexity_score < recursive.complexity_score

    def test_simple_lower_than_deeply_nested(self):
        simple = analyze_schema(SimpleModel)
        deep = analyze_schema(DeeplyNested)
        assert simple.complexity_score < deep.complexity_score


class TestTokenOverhead:
    def test_estimated_tokens_proportional_to_fields(self):
        simple = analyze_schema(SimpleModel)
        wide = analyze_schema(WideModel)
        assert wide.estimated_token_overhead > simple.estimated_token_overhead

    def test_empty_model_zero_overhead(self):
        result = analyze_schema(EmptyModel)
        assert result.estimated_token_overhead == 0


class TestWarningEmission:
    def test_warn_flag_emits_python_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyze_schema(WideModel, warn=True)
            assert len(w) > 0
            assert any("[instructor]" in str(warning.message) for warning in w)

    def test_no_warnings_for_simple_model(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyze_schema(SimpleModel, warn=True)
            assert len(w) == 0


class TestSchemaFinding:
    def test_finding_has_all_fields(self):
        finding = SchemaFinding(
            severity=Severity.WARNING,
            code="TEST",
            message="test message",
            path="$.field",
            suggestion="do something",
        )
        assert finding.severity == Severity.WARNING
        assert finding.code == "TEST"
        assert finding.path == "$.field"

    def test_severity_enum_values(self):
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"


class TestSchemaAnalysisProperties:
    def test_has_errors_false_for_simple(self):
        result = analyze_schema(SimpleModel)
        assert not result.has_errors

    def test_has_errors_true_for_huge_enum(self):
        result = analyze_schema(ModelWithHugeEnum)
        assert result.has_errors

    def test_has_warnings_false_for_simple(self):
        result = analyze_schema(SimpleModel)
        assert not result.has_warnings

    def test_has_warnings_true_for_wide(self):
        result = analyze_schema(WideModel)
        assert result.has_warnings
