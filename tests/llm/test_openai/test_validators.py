from itertools import product
import pytest

import instructor

from typing import Annotated
from pydantic import BaseModel, AfterValidator, BeforeValidator, ValidationError

from instructor.validation import llm_validator
from .util import models, modes


def test_patch_completes_successfully(client):
    class Response(BaseModel):
        message: Annotated[
            str, AfterValidator(instructor.openai_moderation(client=client))
        ]

    with pytest.raises(ValidationError):
        Response(message="I want to make them suffer the consequences")


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_runmodel_validator_error(model, mode, client):
    client = instructor.from_openai(client, mode=mode)

    if mode == instructor.Mode.TOOLS_STRICT:
        # TODO: Structured outputs currently doesn't support the concept of Validators ( This is Pydantic specific ) so perhaps come back to this later
        pytest.skip("Skipping test for structured output")

    class QuestionAnswerNoEvil(BaseModel):
        question: str
        answer: Annotated[
            str,
            BeforeValidator(
                llm_validator(
                    "don't say objectionable things", model=model, client=client
                )
            ),
        ]

    with pytest.raises(ValidationError):
        QuestionAnswerNoEvil(
            question="What is the meaning of life?",
            answer="The meaning of life is to be evil and steal",
        )


@pytest.mark.parametrize("model", models)
def test_runmodel_validator_default_openai_client(model, client):
    client = instructor.from_openai(client)

    class QuestionAnswerNoEvil(BaseModel):
        question: str
        answer: Annotated[
            str,
            BeforeValidator(
                llm_validator(
                    "don't say objectionable things", model=model, client=client
                )
            ),
        ]

    with pytest.raises(ValidationError):
        QuestionAnswerNoEvil(
            question="What is the meaning of life?",
            answer="The meaning of life is to be evil and steal",
        )
