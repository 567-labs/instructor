from __future__ import annotations

import json
from copy import deepcopy
from enum import Enum
from typing import Annotated, Any, Literal, Union, cast

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    create_model,
    field_validator,
)

from instructor.decisions import Choice, Choices, Level, Noul, Question, Score
from instructor.decisions.schema import build_questions, parse_answers


class Action(Choices):
    """Select an action for {{ item }}."""

    ALLOW = Choice(description={"rule": ["Safe"]}, examples=["Hello", 3])
    REVIEW = Choice(value="manual", description="Review")


class PlainAction(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"


def decision_model(annotation: Any, **field_options: Any) -> type[BaseModel]:
    return create_model(
        "Decision", value=(annotation, Field(description="Decide", **field_options))
    )


def reply(kind: str, value: Any) -> dict[str, Any]:
    return {"answers": {"value": {"type": kind, kind: value}}}


def test_choices_enum_values_descriptions_examples_and_typed_results() -> None:
    class Model(BaseModel):
        value: Action

    fields = build_questions(Model, {"item": "message"})
    assert fields[0].question == {
        "type": "choice",
        "instructions": "Select an action for message.",
        "criteria": {
            "allow": {"meaning": {"rule": ["Safe"]}, "examples": ["Hello", "3"]},
            "manual": "Review",
        },
    }
    assert Action("manual") is Action.REVIEW
    result = parse_answers(Model, fields, reply("choice", "manual"), {}, True)
    assert result.value is Action.REVIEW


@pytest.mark.parametrize(
    "annotation,criteria,value",
    [
        (Literal["yes", "no"], {"yes": None, "no": None}, "no"),
        (PlainAction, {"allow": None, "review": None}, PlainAction.REVIEW),
        (
            Annotated[Literal["yes", "no"], Choice(description=["A rule"])],
            {"yes": ["A rule"], "no": ["A rule"]},
            "yes",
        ),
        (
            Union[
                Annotated[Literal["yes"], Choice(description="Accept")],
                Annotated[
                    Literal["no"], Choice(description="Reject", examples=[False])
                ],
            ],
            {"yes": "Accept", "no": {"meaning": "Reject", "examples": ["False"]}},
            "no",
        ),
    ],
)
def test_choice_annotations_generate_criteria_and_validate_results(
    annotation: Any, criteria: dict[str, Any], value: Any
) -> None:
    model = decision_model(annotation)
    fields = build_questions(model, {})
    assert fields[0].question == {
        "type": "choice",
        "instructions": "Decide",
        "criteria": criteria,
    }
    wire_value = value.value if isinstance(value, Enum) else value
    result = parse_answers(model, fields, reply("choice", wire_value), {}, True)
    assert result.model_dump()["value"] == value


def test_question_overrides_field_description_and_nested_templates_do_not_mutate() -> (
    None
):
    instructions = {
        "question": "Choose for {{ item.name }}",
        "rules": ["{{ rule }}", {"fixed": 3.5, "enabled": True, "empty": None}],
    }
    context = {"item": {"name": "post", "confidence": 0.9}, "rule": "Be civil"}
    before = deepcopy((instructions, context))
    model = decision_model(Annotated[Literal["ok"], Question(instructions)])
    first = build_questions(model, context)
    assert first[0].question["instructions"] == {
        "question": "Choose for post",
        "rules": ["Be civil", {"fixed": 3.5, "enabled": True, "empty": None}],
    }
    assert json.loads(json.dumps(first[0].question)) == first[0].question
    first[0].question["instructions"]["rules"].append("changed")
    assert (instructions, context) == before
    assert (
        build_questions(model, {**context, "rule": "Be fair"})[0].question[
            "instructions"
        ]["rules"][0]
        == "Be fair"
    )


@pytest.mark.parametrize("kind", ["choice", "noul", "score"])
@pytest.mark.parametrize("examples", [[], ["Example"]])
def test_generated_criteria_are_independent_of_annotation_data(
    kind: str, examples: list[str]
) -> None:
    meaning = {"rules": ["Original"]}
    original_examples = examples.copy()
    if kind == "choice":
        annotation = Annotated[
            Literal["ok"], Choice(description=meaning, examples=examples)
        ]
    elif kind == "noul":
        annotation = Annotated[
            float,
            Noul(
                "Decide",
                when_true=meaning,
                examples=[(item, True) for item in examples],
            ),
        ]
    else:
        annotation = Annotated[
            float, Score("Decide", levels=[Level(meaning, examples=examples), "High"])
        ]
    model = decision_model(annotation)
    first = build_questions(model, {})
    criteria = first[0].question["criteria"]
    entry = (
        criteria[0]
        if kind == "score"
        else criteria["ok" if kind == "choice" else "true"]
    )
    if examples:
        entry["examples"].append("Changed example")
        entry = entry["meaning"]
    entry["rules"].append("Changed")
    assert meaning == {"rules": ["Original"]}
    assert examples == original_examples
    assert build_questions(model, {})[0].question["criteria"] != criteria


def test_noul_and_score_group_examples_without_modifying_specs() -> None:
    noul = Noul("Flag {{ item }}?", examples=[("good", False), (7, True)])
    score = Score(
        "Rate {{ item }}",
        levels=[Level("Low", examples=["base"]), Level({"rule": "High"})],
        examples=[("extra", 0), (3, 1)],
        scale=(-5, 5),
    )
    before = deepcopy((noul, score))
    model = create_model(
        "Ratings",
        flag=(Annotated[float, noul], ...),
        rating=(Annotated[float, score], ...),
    )
    fields = build_questions(model, {"item": "post"})
    assert fields[0].question == {
        "type": "noul",
        "instructions": "Flag post?",
        "criteria": {
            "true": {"meaning": None, "examples": ["7"]},
            "false": {"meaning": None, "examples": ["good"]},
        },
    }
    assert fields[1].question == {
        "type": "score",
        "instructions": "Rate post",
        "criteria": [
            {"meaning": "Low", "examples": ["base", "extra"]},
            {"meaning": {"rule": "High"}, "examples": ["3"]},
        ],
    }
    assert (noul, score) == before


@pytest.mark.parametrize(
    "spec,kind,wire,expected",
    [
        (Noul("Flag?"), "noul", 0, 0.0),
        (Noul("Flag?"), "noul", 0.5, 0.5),
        (Noul("Flag?"), "noul", 1, 1.0),
        (Score("Rate", ["Low", "Medium", "High"]), "score", 0, 0.0),
        (Score("Rate", ["Low", "Medium", "High"]), "score", 1.5, 1.5),
        (Score("Rate", ["Low", "Medium", "High"]), "score", 2, 2.0),
        (Score("Rate", ["Low", "Medium", "High"], scale=(-10, 10)), "score", 0, -10.0),
        (Score("Rate", ["Low", "Medium", "High"], scale=(-10, 10)), "score", 1.5, 5.0),
        (Score("Rate", ["Low", "Medium", "High"], scale=(-10, 10)), "score", 2, 10.0),
        (Score("Rate", ["Low", "High"], scale=(-1e308, 1e308)), "score", 0, -1e308),
        (Score("Rate", ["Low", "High"], scale=(-1e308, 1e308)), "score", 0.5, 0.0),
        (Score("Rate", ["Low", "High"], scale=(-1e308, 1e308)), "score", 1, 1e308),
    ],
)
def test_numeric_boundaries_and_score_scaling(
    spec: Noul | Score, kind: str, wire: float, expected: float
) -> None:
    model = decision_model(Annotated[float, spec])
    fields = build_questions(model, {})
    raw = reply(kind, wire)
    before = deepcopy(raw)
    result = parse_answers(model, fields, raw, {}, True)
    assert result.model_dump()["value"] == pytest.approx(expected)
    assert raw == before


@pytest.mark.parametrize(
    "kind,spec", [("noul", Noul("Flag?")), ("score", Score("Rate", ["Low", "High"]))]
)
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        "0.5",
        [],
        {},
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_numeric_answers_reject_booleans_nonfinite_and_out_of_range_values(
    kind: str, spec: Noul | Score, value: Any
) -> None:
    model = decision_model(Annotated[float, spec])
    with pytest.raises(ValueError, match=f"value: invalid {kind} answer"):
        parse_answers(model, build_questions(model, {}), reply(kind, value), {}, False)


@pytest.mark.parametrize("raw", [None, [], {}, {"answers": None}, {"answers": []}])
def test_missing_answers_object_is_rejected(raw: Any) -> None:
    model = decision_model(Literal["ok"])
    with pytest.raises(ValueError, match="must contain an answers object"):
        parse_answers(model, build_questions(model, {}), raw, {}, True)


@pytest.mark.parametrize(
    "answer", [None, {}, [], {"type": "score", "score": 0}, {"choice": "ok"}]
)
def test_empty_missing_and_mismatched_field_answers_are_rejected(answer: Any) -> None:
    model = decision_model(Literal["ok"])
    fields = build_questions(model, {})
    for raw in ({"answers": {}}, {"answers": {"value": answer}}):
        with pytest.raises(
            ValueError, match="value: missing or mismatched decision answer"
        ):
            parse_answers(model, fields, raw, {}, True)


@pytest.mark.parametrize("value", [None, "", "unknown", 1, True, [], {}])
def test_choice_answers_require_a_declared_string_option(value: Any) -> None:
    model = decision_model(Literal["ok"])
    with pytest.raises(ValueError, match="value: invalid choice answer"):
        parse_answers(
            model, build_questions(model, {}), reply("choice", value), {}, False
        )


@pytest.mark.parametrize(
    "config,options,input_name",
    [
        ({}, {"alias": "wire"}, "wire"),
        ({}, {"alias": ""}, ""),
        ({}, {"alias": "wire", "validation_alias": ""}, ""),
        ({"populate_by_name": True}, {"alias": "wire"}, "wire"),
        ({}, {"alias": "wire", "validation_alias": "incoming"}, "incoming"),
        ({}, {"serialization_alias": "outgoing"}, "value"),
        pytest.param(
            {"validate_by_alias": False, "validate_by_name": True},
            {"alias": "wire", "validation_alias": "incoming"},
            "value",
            marks=pytest.mark.skipif(
                "validate_by_alias" not in ConfigDict.__annotations__,
                reason="Pydantic before 2.11 does not support validate_by_alias",
            ),
        ),
        pytest.param(
            {"validate_by_alias": False},
            {"alias": "wire"},
            "value",
            marks=pytest.mark.skipif(
                "validate_by_alias" not in ConfigDict.__annotations__,
                reason="Pydantic before 2.11 does not support validate_by_alias",
            ),
        ),
    ],
)
def test_aliases_follow_pydantic_input_configuration_but_keep_question_names(
    config: dict[str, Any], options: dict[str, Any], input_name: str
) -> None:
    model = create_model(
        "Aliased",
        __config__=ConfigDict(**config),
        value=(Literal["ok"], Field(description="Decide", **options)),
    )
    fields = build_questions(model, {})
    assert fields[0].name == "value"
    assert fields[0].input_name == input_name
    assert parse_answers(
        model, fields, reply("choice", "ok"), {}, True
    ).model_dump() == {"value": "ok"}
    with pytest.raises(ValueError, match="missing or mismatched"):
        parse_answers(
            model,
            fields,
            {"answers": {"wire": {"type": "choice", "choice": "ok"}}},
            {},
            True,
        )


@pytest.mark.skipif(
    "validate_by_alias" in ConfigDict.__annotations__,
    reason="Only Pydantic before 2.11 ignores validate_by_alias",
)
def test_legacy_pydantic_configuration_ignores_unsupported_alias_flag() -> None:
    model = create_model(
        "LegacyAlias",
        __config__=ConfigDict(**{"validate_by_alias": False}),
        value=(Literal["ok"], Field(alias="wire", description="Decide")),
    )
    fields = build_questions(model, {})
    assert fields[0].input_name == "wire"
    assert parse_answers(
        model, fields, reply("choice", "ok"), {}, True
    ).model_dump() == {"value": "ok"}


def test_context_and_strictness_reach_real_pydantic_validation() -> None:
    class Model(BaseModel):
        value: Annotated[float, Noul("Flag?")]

        @field_validator("value", mode="before")
        @classmethod
        def apply_context(cls, value: float, info: ValidationInfo) -> str:
            if info.context is None or info.context.get("allow") is not True:
                raise ValueError("context must allow the decision")
            return str(value)

    fields = build_questions(Model, {})
    raw = reply("noul", 0.5)
    with pytest.raises(ValidationError, match="context must allow"):
        parse_answers(Model, fields, raw, {}, False)
    assert parse_answers(Model, fields, raw, {"allow": True}, False).value == 0.5
    with pytest.raises(ValidationError, match="valid number"):
        parse_answers(Model, fields, raw, {"allow": True}, True)


@pytest.mark.parametrize(
    "annotation,message",
    [
        (
            Annotated[float, Noul("Flag"), Score("Rate", ["Low", "High"])],
            "only one decision annotation",
        ),
        (Annotated[float, Noul("Flag"), Choice()], "Choice cannot be combined"),
        (
            Annotated[float, Score("Rate", ["Low", "High"]), Choice()],
            "Choice cannot be combined",
        ),
        (
            Annotated[Literal["ok"], Choice(), Choice(description="Other")],
            "only one Choice annotation",
        ),
        (
            Annotated[Literal["ok"], Question("First"), Question("Second")],
            "only one decision annotation",
        ),
    ],
)
def test_conflicting_annotations_are_rejected(annotation: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_questions(decision_model(annotation), {})


@pytest.mark.parametrize(
    "annotation,message",
    [
        (str, "Expected a string Literal"),
        (Literal[1], "unique nonempty strings"),
        (Literal[""], "unique nonempty strings"),
        (Union[Literal["ok"], None], "Expected a string Literal"),
        (Annotated[str, Choice()], "Inline choices require"),
        (
            Annotated[Literal["ok"], Choice(value="other")],
            "only supported in Choices enums",
        ),
        (
            Union[
                Annotated[Literal["ok"], Choice(description="First")],
                Annotated[Literal["ok"], Choice(description="Second")],
            ],
            "unique nonempty strings",
        ),
        (Annotated[int, Noul("Flag")], "Noul requires a float field"),
        (
            Annotated[int, Score("Rate", ["Low", "High"])],
            "Score requires a float field",
        ),
        (Annotated[float, Score("Rate", [])], "2 to 10 levels"),
        (Annotated[float, Score("Rate", ["Only"])], "2 to 10 levels"),
        (Annotated[float, Score("Rate", ["Level"] * 11)], "2 to 10 levels"),
        (Annotated[float, Score("Rate", [Level(None), "High"])], "require a meaning"),
        (
            Annotated[float, Noul("Flag", examples=cast(Any, [("x", 1)]))],
            "labels must be booleans",
        ),
        (
            Annotated[float, Noul("Flag", when_true="yes", criteria={"true": "yes"})],
            "criteria cannot be combined",
        ),
        (
            Annotated[float, Noul("Flag", criteria={"unknown": "yes"})],
            "only 'true' and 'false'",
        ),
    ],
)
def test_invalid_question_schemas_fail_before_parsing(
    annotation: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_questions(decision_model(annotation), {})


@pytest.mark.parametrize(
    "scale",
    [
        (1, 1),
        (2, 1),
        (True, 2),
        (0, False),
        (0, float("inf")),
        (float("nan"), 1),
        ("0", 1),
        (0,),
        (0, 1, 2),
    ],
)
def test_score_scale_requires_two_finite_increasing_numeric_bounds(scale: Any) -> None:
    model = decision_model(
        Annotated[float, Score("Rate", ["Low", "High"], scale=scale)]
    )
    with pytest.raises(ValueError, match="finite and increasing"):
        build_questions(model, {})


@pytest.mark.parametrize("index", [True, False, -1, 2, 0.5, "0"])
def test_score_example_index_requires_an_in_range_integer(index: Any) -> None:
    model = decision_model(
        Annotated[float, Score("Rate", ["Low", "High"], examples=[("x", index)])]
    )
    with pytest.raises(ValueError, match="invalid Score example level index"):
        build_questions(model, {})


@pytest.mark.parametrize(
    "value",
    [
        {1: "bad"},
        {"nested": [float("nan")]},
        {"nested": {1, 2}},
        {"tuple": (1, 2)},
        {"object": object()},
    ],
)
def test_context_must_be_losslessly_json_compatible(value: Any) -> None:
    with pytest.raises(ValueError, match="context must contain JSON-compatible values"):
        build_questions(decision_model(Literal["ok"]), value)


@pytest.mark.parametrize(
    "instructions", [None, 1, True, "", [], {}, {"bad": float("inf")}, {1: "bad"}]
)
def test_instructions_must_be_nonempty_json_strings_objects_or_arrays(
    instructions: Any,
) -> None:
    model = decision_model(Annotated[Literal["ok"], Question(instructions)])
    with pytest.raises(
        ValueError, match="Instructions and criteria|require instructions"
    ):
        build_questions(model, {})


@pytest.mark.parametrize("meaning", [1, True, {"bad": float("nan")}, {1: "bad"}])
def test_choice_criteria_reject_non_json_or_non_structured_meanings(
    meaning: Any,
) -> None:
    with pytest.raises(ValueError, match="Instructions and criteria"):
        build_questions(
            decision_model(Annotated[Literal["ok"], Choice(description=meaning)]), {}
        )


def test_empty_model_invalid_model_and_non_dictionary_context_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one question"):
        build_questions(create_model("Empty"), {})
    for invalid in (None, object, decision_model(Literal["ok"])(value="ok")):
        with pytest.raises(TypeError, match="Pydantic BaseModel"):
            build_questions(cast(Any, invalid), {})
    with pytest.raises(TypeError, match="context must be a dictionary"):
        build_questions(decision_model(Literal["ok"]), cast(Any, []))


@pytest.mark.parametrize(
    "alias", [AliasPath("nested", "value"), AliasChoices("a", "b")]
)
def test_complex_validation_aliases_are_rejected(alias: Any) -> None:
    with pytest.raises(ValueError, match="complex validation aliases"):
        build_questions(decision_model(Literal["ok"], validation_alias=alias), {})


def test_colliding_validation_input_names_are_rejected() -> None:
    model = create_model(
        "Colliding",
        first=(Literal["ok"], Field(alias="same", description="First")),
        second=(Literal["ok"], Field(validation_alias="same", description="Second")),
    )
    with pytest.raises(ValueError, match="validation aliases must be unique"):
        build_questions(model, {})


@pytest.mark.parametrize("count", [1, 255])
def test_choice_option_count_accepts_both_boundaries(count: int) -> None:
    annotation = cast(Any, Literal)[tuple(f"option-{index}" for index in range(count))]
    fields = build_questions(decision_model(annotation), {})
    assert len(fields[0].question["criteria"]) == count
    assert fields[0].question["criteria"][f"option-{count - 1}"] is None


def test_choice_option_count_rejects_more_than_255() -> None:
    annotation = cast(Any, Literal)[tuple(f"option-{index}" for index in range(256))]
    with pytest.raises(ValueError, match="between 1 and 255"):
        build_questions(decision_model(annotation), {})


def test_choices_declarations_reject_non_choices_and_duplicate_values() -> None:
    with pytest.raises(TypeError, match="must be a Choice"):

        class Invalid(Choices):
            BAD = "not a Choice"

    with pytest.raises(ValueError, match="unique nonempty strings"):

        class Duplicate(Choices):
            FIRST = Choice(value="same")
            SECOND = Choice(value="same")


@pytest.mark.parametrize("value", ["", 1, False])
def test_choices_declarations_reject_invalid_wire_values(value: Any) -> None:
    with pytest.raises(ValueError, match="unique nonempty strings"):

        class Invalid(Choices):
            BAD = Choice(value=value)


def test_plain_enums_reject_non_string_values_aliases_and_empty_options() -> None:
    class Numeric(Enum):
        FIRST = 1

    class Aliased(Enum):
        FIRST = "same"
        SECOND = "same"

    class Empty(Enum):
        pass

    for enum in (Numeric, Aliased):
        with pytest.raises(ValueError, match="unique nonempty strings"):
            build_questions(decision_model(enum), {})
    with pytest.raises(ValueError, match="between 1 and 255"):
        build_questions(decision_model(Empty), {})


def test_score_accepts_ten_levels_and_uses_the_last_example_index() -> None:
    spec = Score("Rate", [str(index) for index in range(10)], examples=[("last", 9)])
    model = decision_model(Annotated[float, spec])
    fields = build_questions(model, {})
    assert fields[0].question["criteria"] == [str(index) for index in range(9)] + [
        {"meaning": "9", "examples": ["last"]}
    ]
    assert parse_answers(model, fields, reply("score", 9), {}, True).model_dump() == {
        "value": 9.0
    }
    with pytest.raises(ValueError, match="invalid score answer"):
        parse_answers(model, fields, reply("score", 9.01), {}, False)


def test_scaled_answers_still_obey_pydantic_field_constraints() -> None:
    model = decision_model(
        Annotated[float, Score("Rate", ["Low", "High"], scale=(0, 10))],
        ge=2,
        le=8,
    )
    fields = build_questions(model, {})
    assert parse_answers(model, fields, reply("score", 0.5), {}, True).model_dump() == {
        "value": 5.0
    }
    for wire in (0, 1):
        with pytest.raises(
            ValidationError, match="less than or equal|greater than or equal"
        ):
            parse_answers(model, fields, reply("score", wire), {}, False)


def test_model_defaults_do_not_replace_missing_decision_answers() -> None:
    model = create_model(
        "Defaulted",
        value=(Literal["ok"], Field(default="ok", description="Decide")),
    )
    with pytest.raises(ValueError, match="missing or mismatched decision answer"):
        parse_answers(model, build_questions(model, {}), {"answers": {}}, {}, True)


def test_async_validators_are_rejected_during_schema_building() -> None:
    from instructor.v2.validation.async_validators import async_field_validator

    class Model(BaseModel):
        value: Literal["ok"] = Field(description="Decide")

        @async_field_validator("value")
        async def validate_value(cls, value: str) -> str:
            return value

    with pytest.raises(ValueError, match="async validators are not supported"):
        build_questions(Model, {})
