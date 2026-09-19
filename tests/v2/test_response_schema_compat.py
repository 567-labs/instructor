"""Compatibility tests for deprecated ResponseSchema provider helpers."""

from __future__ import annotations

from typing import Annotated, Any, cast

import pytest
from pydantic import Field, ValidationError
from typing_extensions import NotRequired, ReadOnly, Required, TypedDict

from instructor import Mode, Provider
from instructor.v2.core.function_calls import ResponseSchema
from instructor.v2.core.response_model import prepare_response_model


class Answer(ResponseSchema):
    answer: float


def test_schema_properties_delegate_to_provider_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, type[Answer]]] = []

    def fake_openai_schema(model: type[Answer]) -> dict[str, str]:
        calls.append(("openai", model))
        return {"provider": "openai"}

    def fake_anthropic_schema(model: type[Answer]) -> dict[str, str]:
        calls.append(("anthropic", model))
        return {"provider": "anthropic"}

    def fake_gemini_schema(model: type[Answer]) -> dict[str, str]:
        calls.append(("gemini", model))
        return {"provider": "gemini"}

    monkeypatch.setattr(
        "instructor.v2.providers.openai.schema.generate_openai_schema",
        fake_openai_schema,
    )
    monkeypatch.setattr(
        "instructor.v2.providers.anthropic.schema.generate_anthropic_schema",
        fake_anthropic_schema,
    )
    monkeypatch.setattr(
        "instructor.v2.providers.gemini.schema.generate_gemini_schema",
        fake_gemini_schema,
    )

    assert Answer.openai_schema == {"provider": "openai"}
    assert Answer.anthropic_schema == {"provider": "anthropic"}
    assert Answer.gemini_schema == {"provider": "gemini"}
    assert calls == [
        ("openai", Answer),
        ("anthropic", Answer),
        ("gemini", Answer),
    ]


@pytest.mark.parametrize(
    ("method_name", "mode", "provider"),
    [
        ("parse_genai_structured_outputs", Mode.JSON, Provider.GENAI),
        ("parse_genai_tools", Mode.TOOLS, Provider.GENAI),
        ("parse_cohere_json_schema", Mode.JSON_SCHEMA, Provider.COHERE),
        ("parse_bedrock_json", Mode.MD_JSON, Provider.BEDROCK),
        ("parse_bedrock_tools", Mode.TOOLS, Provider.BEDROCK),
        ("parse_gemini_json", Mode.MD_JSON, Provider.GEMINI),
        ("parse_gemini_tools", Mode.TOOLS, Provider.GEMINI),
        ("parse_vertexai_tools", Mode.TOOLS, Provider.VERTEXAI),
        ("parse_vertexai_json", Mode.MD_JSON, Provider.VERTEXAI),
        ("parse_cohere_tools", Mode.TOOLS, Provider.COHERE),
        ("parse_writer_tools", Mode.TOOLS, Provider.WRITER),
        ("parse_writer_json", Mode.MD_JSON, Provider.WRITER),
        ("parse_mistral_structured_outputs", Mode.JSON_SCHEMA, Provider.MISTRAL),
    ],
)
def test_provider_parse_helpers_delegate_to_registry(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    mode: Mode,
    provider: Provider,
) -> None:
    calls: list[dict[str, Any]] = []
    sentinel = object()

    def fake_parse_with_registry(
        cls: type[ResponseSchema],
        completion: Any,
        *,
        mode: Mode,
        provider: Provider,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        warning: str | None = None,
    ) -> object:
        calls.append(
            {
                "cls": cls,
                "completion": completion,
                "mode": mode,
                "provider": provider,
                "validation_context": validation_context,
                "strict": strict,
                "warning": warning,
            }
        )
        return sentinel

    monkeypatch.setattr(
        Answer,
        "_parse_with_registry",
        classmethod(fake_parse_with_registry),
    )

    completion = object()
    kwargs: dict[str, Any] = {"validation_context": {"source": "compat"}}
    if method_name != "parse_vertexai_tools":
        kwargs["strict"] = True
    result = getattr(Answer, method_name)(completion, **kwargs)

    assert result is sentinel
    assert calls == [
        {
            "cls": Answer,
            "completion": completion,
            "mode": mode,
            "provider": provider,
            "validation_context": {"source": "compat"},
            "strict": False if method_name == "parse_vertexai_tools" else True,
            "warning": calls[0]["warning"],
        }
    ]
    assert calls[0]["warning"] is not None


class OptionalUser(TypedDict):
    name: str
    age: NotRequired[int]


class PartialUser(TypedDict, total=False):
    nickname: str
    user_id: Required[int]


def test_prepare_response_model_preserves_typed_dict_key_semantics() -> None:
    model = cast(Any, prepare_response_model(OptionalUser))

    assert model.model_fields["name"].is_required()
    assert not model.model_fields["age"].is_required()
    assert model(name="Ada").model_dump(exclude_unset=True) == {"name": "Ada"}

    partial_model = cast(Any, prepare_response_model(PartialUser))
    assert not partial_model.model_fields["nickname"].is_required()
    assert partial_model.model_fields["user_id"].is_required()


def test_prepare_response_model_preserves_iterable_typed_dict_keys() -> None:
    iterable_model = cast(Any, prepare_response_model(list[OptionalUser]))
    task_model = iterable_model.task_type

    assert task_model.model_fields["name"].is_required()
    assert not task_model.model_fields["age"].is_required()
    assert task_model(name="Ada").model_dump(exclude_unset=True) == {"name": "Ada"}


@pytest.mark.parametrize("total", [True, False])
@pytest.mark.parametrize("as_list", [True, False])
@pytest.mark.parametrize(
    ("annotation", "required"),
    [
        (ReadOnly[int], None),
        (Required[ReadOnly[int]], True),
        (ReadOnly[Required[int]], True),
        (NotRequired[ReadOnly[int]], False),
        (ReadOnly[NotRequired[int]], False),
        (ReadOnly[Annotated[int, Field(gt=0)]], None),
    ],
)
def test_prepare_response_model_preserves_readonly_typed_dict_fields(
    annotation: Any, required: bool | None, total: bool, as_list: bool
) -> None:
    make_typed_dict = cast(Any, TypedDict)
    record = make_typed_dict("Record", {"value": annotation}, total=total)
    model = cast(Any, prepare_response_model(list[record] if as_list else record))
    if as_list:
        model = model.task_type
    is_required = total if required is None else required

    assert model.model_fields["value"].is_required() is is_required
    schema = model.model_json_schema()
    assert schema["properties"]["value"]["type"] == "integer"
    assert ("value" in schema.get("required", [])) is is_required
    assert model.model_validate_json('{"value": 3}').value == 3
    with pytest.raises(ValidationError):
        model.model_validate_json('{"value": "invalid"}')

    if is_required:
        with pytest.raises(ValidationError):
            model.model_validate_json("{}")
    else:
        assert model.model_validate_json("{}").model_dump(exclude_unset=True) == {}


def test_prepare_response_model_preserves_readonly_field_metadata() -> None:
    class Record(TypedDict):
        value: Annotated[
            ReadOnly[Annotated[int, Field(gt=0, description="Inner count")]],
            Field(lt=10, description="A count"),
        ]

    model = cast(Any, prepare_response_model(Record))
    schema = model.model_json_schema()["properties"]["value"]
    assert schema["exclusiveMinimum"] == 0
    assert schema["exclusiveMaximum"] == 10
    assert schema["description"] == "A count"
    assert model.model_validate_json('{"value": 3}').value == 3
    for value in (0, 10):
        with pytest.raises(ValidationError):
            model.model_validate({"value": value})
