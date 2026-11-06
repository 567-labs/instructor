"""Structured-output validators."""

from typing import Optional

from pydantic import Field

from instructor.v2.core.function_calls import ResponseSchema


class Validator(ResponseSchema):
    """Describe whether a candidate attribute is valid and how to repair it."""

    is_valid: bool = Field(
        description="Whether the attribute is valid based on the requirements",
    )
    reason: Optional[str] = Field(
        default=None,
        description="The error message if the attribute is not valid, otherwise None",
    )
    fixed_value: Optional[str] = Field(
        default=None,
        description="If the attribute is not valid, suggest a new value for the attribute",
    )
