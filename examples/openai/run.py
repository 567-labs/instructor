"""
Canonical OpenAI starter example for the instructor library.

Demonstrates how to use `instructor.from_provider()` with OpenAI to extract
structured data from natural language into a Pydantic model.

Usage:
    export OPENAI_API_KEY=your-api-key
    python examples/openai/run.py
"""

import instructor
from pydantic import BaseModel, Field


class UserInfo(BaseModel):
    """Extracted user information."""

    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")


client = instructor.from_provider("openai/gpt-4o-mini")

user = client.chat.completions.create(
    response_model=UserInfo,
    messages=[
        {
            "role": "user",
            "content": "Extract: Jason is 25 years old.",
        }
    ],
)

print(user.model_dump_json(indent=2))
