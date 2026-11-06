import pytest
from pydantic import BaseModel, field_validator
import instructor
from .util import models


@pytest.mark.parametrize("mode", [instructor.Mode.GENAI_TOOLS])
def test_genai_tools_validation_retry_preserves_model_content(mode):
    """Ensure GENAI_TOOLS validation retries are wired end-to-end."""
    from instructor.core.exceptions import InstructorRetryException

    model = models[0]

    class AlwaysInvalid(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def always_fail(cls, v: int) -> int:  # noqa: ARG003
            raise ValueError("force retry for reask validation coverage")

    client = instructor.from_provider(f"google/{model}", mode=mode)
    with pytest.raises(InstructorRetryException) as exc_info:
        client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Return any integer value",
                }
            ],
            response_model=AlwaysInvalid,
            max_retries=2,
        )

    # max_retries excludes the initial attempt.
    assert exc_info.value.n_attempts == 3
