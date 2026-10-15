"""Compatibility tests for deprecated ResponseSchema provider helpers."""

from __future__ import annotations

from typing import Any, cast

from pydantic import ValidationError
import pytest
from typing_extensions import NotRequired, Required, TypedDict

from instructor import Mode, Provider
from instructor.v2.core.function_calls import ResponseSchema
from instructor.v2.core.response_model import prepare_response_model
from instructor.processing.response import handle_response_model


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


class UnderscoreKeyRecord(TypedDict):
    _id: str
    typed_dict_field_0: str
    name: str


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


def test_typed_dict_underscore_keys_are_schema_fields_and_validate_by_alias() -> None:
    model, request = handle_response_model(UnderscoreKeyRecord)
    schema = request["tools"][0]["function"]["parameters"]

    assert schema["properties"]["_id"]["type"] == "string"
    assert set(schema["required"]) == {"_id", "typed_dict_field_0", "name"}

    parsed = model.model_validate(
        {"_id": "doc-1", "typed_dict_field_0": "collision-safe", "name": "Report"}
    )
    assert parsed.model_dump(by_alias=True) == {
        "_id": "doc-1",
        "typed_dict_field_0": "collision-safe",
        "name": "Report",
    }

    with pytest.raises(ValidationError):
        model.model_validate({"typed_dict_field_0": "collision-safe", "name": "Report"})
