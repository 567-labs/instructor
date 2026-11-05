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

import os

import instructor
from pydantic import BaseModel, Field, ValidationInfo, model_validator


class Claim(BaseModel):
    """A single atomic claim extracted from a larger piece of text."""

    text: str = Field(description="The claim, stated as a short standalone sentence.")
    quote: str = Field(description="Exact supporting words copied from the input text.")
    is_verifiable: bool = Field(
        description=(
            "True if the claim is a factual statement that can be checked "
            "against a source. False if it is an opinion or subjective."
        )
    )


class ClaimList(BaseModel):
    """A list of atomic claims extracted from the input text."""

    claims: list[Claim]

    @model_validator(mode="after")
    def check_quotes(self, info: ValidationInfo):
        source = (info.context or {}).get("source")
        if not isinstance(source, str):
            raise ValueError("Provide the original text in validation context")
        for claim in self.claims:
            if not claim.quote.strip() or claim.quote not in source:
                raise ValueError("Each claim needs an exact quote from the input text")
        return self


def extract_claims(text: str) -> ClaimList:
    """Break a piece of text into a list of atomic, labelled claims."""
    client = instructor.from_provider("groq/" + os.environ["GROQ_MODEL"])
    return client.chat.completions.create(
        context={"source": text},
        response_model=ClaimList,
        messages=[
            {
                "role": "user",
                "content": f"Break the following text into individual claims: {text}",
            }
        ],
    )


if __name__ == "__main__":
    statement = "The Vega bottle holds 750 mL. Its finish is beautiful."

    result = extract_claims(statement)

    for i, claim in enumerate(result.claims, start=1):
        print(f"{i}. {claim.text} -> verifiable: {claim.is_verifiable}")
