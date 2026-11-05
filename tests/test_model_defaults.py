"""Offline regression tests for public model defaults."""

from __future__ import annotations

from inspect import signature

from instructor.distil import Instructions
from instructor.v2.validation.llm_validators import llm_validator


CURRENT_DEFAULT_MODEL = "gpt-4.1-mini"


def test_llm_validator_uses_current_default_model() -> None:
    assert signature(llm_validator).parameters["model"].default == CURRENT_DEFAULT_MODEL


def test_distil_uses_current_default_model() -> None:
    assert (
        signature(Instructions.distil).parameters["model"].default
        == CURRENT_DEFAULT_MODEL
    )
