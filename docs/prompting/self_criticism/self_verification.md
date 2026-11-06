---
title: "Self-Verify LLM Responses"
description: "The self-verification framework generates multiple response candidates, then uses an LLM to verify these candidates."
---

We want to verify that an LLM response is correct. How can we automate this?

The self-verification framework generates multiple response candidates, then uses an LLM to verify these candidates. The process follows two stages:

1. Forward Reasoning
2. Backward Verification

## Forward Reasoning
In forward reasoning, we leaverage CoT to generate multiple candidate solutions.

## Backward Verification
Backward verification involves three steps.

### Rewrite As Declarative

Rewrite the original question and its solution as a declarative.

!!! example "Rewritten Declarative Example"
    **original question**: Jackie has 10 apples. Adam has 8 apples. How many more apples does Jackie have than Adam?
    **response candidate**: Jackie has 10 apples. so Jackie has 10-8=2 more apples than Adam, and the answer is 2.
    **rewritten declarative**: Jackie has 10 apples. Adam has 8 apples. Jackie has 2 more apples than Adam.

### Construct New Question

Construct a new question and prompt the LLM to verify it. Two possible methods are:

1. True-False Item Verification (TFV)
2. Condition Mask Verification (CMV)

TFV asks the LLM if the rewritten declarative is correct. CMV filters out conditions provided in the original question and asks an LLM to predict the filtered condition.

!!! example "TFV Example Prompt"
    Jackie has 10 apples. Adam has 8 apples. Jackie has 2 more apples than Adam. Is this correct?

!!! example "CMV Example Prompt"
    Jackie has X apples. Adam has 8 apples. Jackie has 2 more apples than Adam. What is X?

### Compute Verification Score
The LLM is then queried with the new question for each candidate *k* times. If TFV is used, the verification score is simply the number of times the LLM outputs "True". If CMV is used, the verification score is the number of times the masked value and the real value match.

The candidate with the highest verification score is then chosen as the final answer.

## Implementation

The full pipeline with forward reasoning and backward verification can be implemented using `instructor` as seen below:

```python
import instructor
from pydantic import BaseModel
from typing import Literal
client = instructor.from_provider("openai/gpt-5-nano")

n = 3  # Number of candidates to generate
k = 5  # Number of times to verify


class Date(BaseModel):
    month: int
    day: int


class Candidate(BaseModel):
    reasoning_steps: list[str]
    month: str


class Rewritten(BaseModel):
    declarative: str


class Verification(BaseModel):
    correct: Literal["True", "False"]


def query_llm(query, model):
    return client.create(
        model="gpt-4o",
        response_model=model,
        messages=[
            {
                "role": "user",
                "content": f"Think step by step: {query}",
            }
        ],
    )


def rewrite(query, candidate):
    return client.create(
        model="gpt-4o",
        response_model=Rewritten,
        messages=[
            {
                "role": "user",
                "content": f"""
                    Please change the questions and answers into complete declarative sentences
                    {query}
                    The answer is {candidate.month}.
                """,
            }
        ],
    )


def verify(question):
    return client.create(
        model="gpt-4o",
        response_model=Verification,
        messages=[{"role": "user", "content": question}],
    )


if __name__ == "__main__":
    query = "What month is it now if it has been 3 weeks, 10 days, and 2 hours since May 1, 2024 6pm?"

    # Step 1: Forward Reasoning
    candidates = [query_llm(query, Candidate) for _ in range(n)]

    # Step 2: Backwards Verification
    for candidate in candidates:
        # 2.a Rewrite
        rewritten = rewrite(query, candidate)
        # 2.b Construct new questions
        question = f"{rewritten.declarative} Do it is correct (True or False)?"
        # 2.c Compute verification score
        scores = [verify(question).correct for _ in range(k)]
        verification_score = sum(1 for s in scores if s == "True")

        print(f"Candidate: {candidate.month}, Verification Score: {verification_score}")
        #> Candidate: May, Verification Score: 0
        #> Candidate: June, Verification Score: 2
        #> Candidate: May, Verification Score: 1
```

### References

<sup id="ref-1">1</sup>: [Large Language Models are Better Reasoners with Self-Verification](https://arxiv.org/abs/2212.09561)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)