import pytest
import instructor
from pydantic import BaseModel


class Answer(BaseModel):
    answer: float


def test_reasoning():
    client = instructor.from_provider(
        "anthropic/claude-sonnet-5",
        mode=instructor.Mode.ANTHROPIC_REASONING_TOOLS,
    )
    try:
        response = client.chat.completions.create(
            response_model=Answer,
            messages=[
                {
                    "role": "user",
                    "content": "Which is larger, 9.11 or 9.8? Think carefully about decimal places.",
                },
            ],
            max_tokens=2000,
            thinking={"type": "adaptive"},
            tool_choice={"type": "auto"},
            max_retries=3,  # Retry if the model gets it wrong
        )
    except Exception as e:
        if "404" in str(e) or "not_found_error" in str(e):
            pytest.skip("Model claude-sonnet-5 not available with current API key")
        raise

    # Assertions to validate the response
    assert isinstance(response, Answer)
    assert response.answer == 9.8
