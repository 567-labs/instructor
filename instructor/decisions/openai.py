"""Translate the shared decision contract to OpenAI's Decisions wire format."""

from __future__ import annotations

import json
from typing import Any

from .schema import _Field


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def request_body(
    model: str, fields: list[_Field], context: dict[str, Any]
) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    for field in fields:
        source = field.question
        question: dict[str, Any] = {
            "name": field.name,
            "type": "predicate" if field.kind == "noul" else field.kind,
            "instructions": _text(source["instructions"]),
        }
        if field.kind == "choice":
            question["choices"] = [
                {
                    "value": value,
                    **({"description": _text(meaning)} if meaning is not None else {}),
                }
                for value, meaning in source["criteria"].items()
            ]
        elif field.kind == "score":
            question["levels"] = [
                {"label": str(index), "description": _text(meaning)}
                for index, meaning in enumerate(source["criteria"])
            ]
        elif "criteria" in source:
            # OpenAI predicates have no separate criteria field.
            question["instructions"] += "\nCriteria: " + _text(source["criteria"])
        questions.append(question)
    return {"model": model, "input": _text(context), "questions": questions}


def answer_envelope(fields: list[_Field], raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), list):
        raise ValueError("OpenAI decision response must contain an answers array")
    answers = raw["answers"]
    if len(answers) != len(fields):
        raise ValueError("OpenAI decision answer count does not match questions")
    normalized = {}
    for field, answer in zip(fields, answers):
        if not isinstance(answer, dict) or answer.get("name") != field.name:
            raise ValueError(f"{field.name}: missing or mismatched OpenAI answer name")
        if answer.get("type") == "refusal":
            raise ValueError(f"{field.name}: OpenAI refused the decision question")
        expected = "predicate" if field.kind == "noul" else field.kind
        if answer.get("type") != expected:
            raise ValueError(f"{field.name}: mismatched OpenAI decision answer type")
        value_key = "probability" if field.kind == "noul" else field.kind
        normalized[field.name] = {"type": field.kind, field.kind: answer.get(value_key)}
    return {"answers": normalized}
