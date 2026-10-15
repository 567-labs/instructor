"""
Claim Extraction Example
========================

This example shows how to use Instructor to decompose a block of text into
individual atomic claims and label each one as verifiable (a factual statement
that can be checked against a source) or not (an opinion or subjective phrase).

This pattern is a useful building block for fact-checking and hallucination
detection pipelines, where an LLM answer is first broken into small claims
before each claim is verified against retrieved evidence.
"""

from typing import List

import instructor
from groq import Groq
from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A single atomic claim extracted from a larger piece of text."""

    text: str = Field(description="The claim, stated as a short standalone sentence.")
    is_verifiable: bool = Field(
        description=(
            "True if the claim is a factual statement that can be checked "
            "against a source. False if it is an opinion or subjective."
        )
    )


class ClaimList(BaseModel):
    """A list of atomic claims extracted from the input text."""

    claims: List[Claim]


# Patch the Groq client so it can return structured Pydantic models.
client = instructor.from_groq(Groq())


def extract_claims(text: str) -> ClaimList:
    """Break a piece of text into a list of atomic, labelled claims."""
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=ClaimList,
        messages=[
            {
                "role": "user",
                "content": f"Break the following text into individual claims: {text}",
            }
        ],
    )


if __name__ == "__main__":
    statement = (
        "The Eiffel Tower is in Paris and it was built in 1889. It is beautiful."
    )

    result = extract_claims(statement)

    for i, claim in enumerate(result.claims, start=1):
        print(f"{i}. {claim.text} -> verifiable: {claim.is_verifiable}")