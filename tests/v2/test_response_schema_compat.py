"""Compatibility tests for deprecated ResponseSchema provider helpers."""

from __future__ import annotations

from typing import Any, cast

import pytest
from typing_extensions import NotRequired, Required, TypedDict

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


class RecordWithUnderscore(TypedDict):
    _id: str
    name: str


class OptionalRecordWithUnderscore(TypedDict, total=False):
    _id: str
    name: str


def test_prepare_response_model_preserves_typed_dict_underscore_keys() -> None:
    from pydantic import ValidationError
    from instructor.processing.response import handle_response_model

    model = cast(Any, prepare_response_model(RecordWithUnderscore))
    schema = model.model_json_schema()
    assert "_id" in schema["properties"]
    assert "_id" in schema["required"]

    validated = model.model_validate({"_id": "doc-1", "name": "Report"})
    assert validated.model_dump(by_alias=True) == {"_id": "doc-1", "name": "Report"}

    with pytest.raises(ValidationError):
        model.model_validate({"name": "Report"})

    # Check handle_response_model integration
    resp_model, request = handle_response_model(
        RecordWithUnderscore,
        mode=Mode.TOOLS,
        messages=[{"role": "user", "content": "Extract"}],
    )
    tools = request["tools"][0]["function"]["parameters"]
    assert "_id" in tools["properties"]
    assert "_id" in tools["required"]

    # Check optional underscore keys
    opt_model = cast(Any, prepare_response_model(OptionalRecordWithUnderscore))
    opt_schema = opt_model.model_json_schema()
    assert "_id" in opt_schema["properties"]
    assert "_id" not in opt_schema.get("required", [])
    assert opt_model.model_validate({"name": "Report"}).model_dump(
        by_alias=True, exclude_unset=True
    ) == {"name": "Report"}
    assert opt_model.model_validate({"_id": "doc-2"}).model_dump(
        by_alias=True, exclude_unset=True
    ) == {"_id": "doc-2"}

    # Check iterable / list of TypedDict with underscore keys
    iterable_model = cast(Any, prepare_response_model(list[RecordWithUnderscore]))
    task_model = iterable_model.task_type
    task_schema = task_model.model_json_schema()
    assert "_id" in task_schema["properties"]
    assert "_id" in task_schema["required"]

