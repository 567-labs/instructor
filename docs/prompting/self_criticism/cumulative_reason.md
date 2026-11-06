---
description: "Cumulative Reasoning breaks the reasoning process into three separate steps so that our model has enough room to reason and filter out the reasoning steps at each point, thus improving model performance"
---

Cumulative Reasoning<sup><a href="https://arxiv.org/pdf/2308.04371">1</a></sup> aims to generate better outputs by dividing the reasoning process into three separate steps

1. **Propose** : A LLM first suggests potential steps based on the current context, initiating the reasoning cycle
2. **Verify** : We then assess the proposer's suggestions for accuracy, incorporating valid steps into the ongoing context
3. **Report** : We then determine the appropriate moment to conclude the reasoning process

By first generating potential steps and separating out each portions of the reasoning process, we are able to obtain significant improvements in logical inference tasks and mathematical problems.

We can implement this using `instructor` as seen below

```python hl_lines="46-61 94-100 138-148"
import instructor
from pydantic import BaseModel, Field
from textwrap import dedent
from typing import Literal
import asyncio
client = instructor.from_provider("openai/gpt-5-nano", async_client=True)


class Proposition(BaseModel):
    premise1: str
    premise2: str
    reasoning: str
    proposition: str


class ProposerOutput(BaseModel):
    reasoning: str
    valid_propositions: list[Proposition] = Field(
        description="Concise list of Propositions that are derived from the premises that are relevant to the hypothesis. Note that each Proposition is derived from two given premises at most",
        min_length=4,
    )
    prediction: Literal["False", "True", "Unknown"]


class VerifiedProposition(BaseModel):
    proposition: str
    reasoning: str
    is_valid: bool


class ReporterOutput(BaseModel):
    reasoning: str
    is_valid_hypothesis: bool


async def generate_propositions(premises: list[str], hypothesis: str) -> ProposerOutput:
    formatted_premises = "\n- ".join(premises)
    return await client.create(
        messages=[
            {
                "role": "system",
                "content": dedent(
                    """
                Suppose you are one of the greatest AI
                scientists, logicians, and mathematicians.

                Let us think step by step. Please use
                First-Order Logic (FOL) to deduce a list
                of Propositions. Each Proposition is
                derived from two given Premises and
                should be logically correct. Most
                importantly, each Proposition should
                not duplicate the two premises that it
                is derived from. Please make sure your
                reasoning is directly deduced from the
                Premises and Propositions rather than
                introducing unsourced common knowledge
                and unsourced information by common
                sense reasoning.
                """
                ),
            },
            {
                "role": "user",
                "content": dedent(
                    f"""
                Premises:
                {formatted_premises}

                We want to deduce more Propositions to
                determine the correctness of the following
                Hypothesis:
                Hypothesis: {hypothesis}
                """
                ),
            },
        ],
        response_model=ProposerOutput,
        model="gpt-4o",
    )


async def verify_propositions(
    premise_evaluation: ProposerOutput,
) -> list[VerifiedProposition]:
    async def create_verification_task(proposition: Proposition) -> VerifiedProposition:
        return await client.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                    Suppose you are one of the greatest AI
                    scientists, logicians, and mathematicians.
                    Let us think step by step. Please use
                    First-Order Logic (FOL) to determine
                    whether the deduction of two given
                    Premises to a Proposition is valid or not,
                    and reply with True or False.
                    """,
                },
                {
                    "role": "user",
                    "content": f"""
                    Premises:
                    {proposition.premise1}
                    {proposition.premise2}

                    Proposition:
                    {proposition.proposition}
                    """,
                },
            ],
            response_model=VerifiedProposition,
            model="gpt-4o",
        )

    tasks = [
        create_verification_task(proposition)
        for proposition in premise_evaluation.valid_propositions
    ]

    return await asyncio.gather(*tasks)


async def final_evaluation(
    verification_result: list[str], hypothesis: str, premises: list[str]
) -> ReporterOutput:
    formatted_premises = "\n- ".join(premises)
    formatted_propositions = "\n- ".join(verification_result)
    return await client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                Suppose you are one of the greatest AI
                scientists, logicians, and mathematicians.
                Let us think step by step. Read and analyze
                the “Premises” first, then use First-Order
                Logic (FOL) to judge whether the “Hypothesis”
                is True, False, or Unknown. Please make sure
                your reasoning is directly deduced from the
                "Premises" and "Propositions" rather than
                introducing unsourced common knowledge and
                unsourced information by common sense
                reasoning.
                """,
            },
            {
                "role": "user",
                "content": f"""
                Premises:
                {formatted_premises}

                Hypothesis: {hypothesis}
                """,
            },
            {
                "role": "assistant",
                "content": f"""
                Let's think step by step. From the premises,
                we can deduce the following propositions:
                {formatted_propositions}

                Recall the Hypothesis: {hypothesis}
                """,
            },
        ],
        response_model=ReporterOutput,
    )


if __name__ == "__main__":
    hypothesis = "Hyraxes lay eggs"
    premises = [
        "The only types of mammals that lay eggs are platypuses and echidnas",
        "Platypuses are not hyrax",
        "Echidnas are not hyrax",
        "No mammals are invertebrates",
        "All animals are either vertebrates or invertebrates",
        "Mammals are animals",
        "Hyraxes are mammals",
        "Grebes lay eggs",
        "Grebes are not platypuses and also not echidnas",
    ]
    premise_evaluation = asyncio.run(generate_propositions(premises, hypothesis))

    verification_result = asyncio.run(verify_propositions(premise_evaluation))

    filtered_propositions = [
        proposition.proposition
        for proposition in verification_result
        if proposition.is_valid
    ]

    reporter_output = asyncio.run(
        final_evaluation(filtered_propositions, hypothesis, premises)
    )
    print(reporter_output.model_dump_json(indent=2))
    """
    {
      "reasoning": "Based on the premises provided, the
      only mammals that lay eggs are platypuses and
      echidnas. Hyraxes are mammals but are explicitly
      stated as not being platypuses or echidnas. Hence,
      there is no basis in the premises to conclude that
      hyraxes lay eggs. \n\nTherefore, the hypothesis that
      hyraxes lay eggs is False.",
      "is_valid_hypothesis": false
    }
    """
```

### References

<sup id="ref-1">1</sup>: [Cumulative Reasoning with Large Language Models](https://arxiv.org/pdf/2308.04371)
