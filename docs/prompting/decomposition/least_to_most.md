---
title: "Solve simpler subproblems"
description: "Least-to-Most is a prompting technique that breaks a complex problem down into a series of increasingly complex subproblems."
---

Given a complex problem, how can we encourage an LLM to solve simpler subproblems?

Least-to-Most is a prompting technique that breaks a complex problem down into a series of increasingly complex subproblems.

!!! example "Subproblems Example"
    **original problem**: Adam is twice as old as Mary. Adam will be 11 in 1 year. How old is Mary?

    **subproblems**: (1) How old is Adam now? (2) What is half of Adam's current age?

These subproblems are solved sequentially, allowing the answers from earlier (simpler) subproblems to inform the LLM while solving later (more complex) subproblems.

```python
import instructor
from pydantic import BaseModel
from typing import Iterable

class Subquestion(BaseModel):
    question: str


class Answer(BaseModel):
    answer: int


class SubquestionWithAnswers(BaseModel):
    question: str
    answer: int


client = instructor.from_provider("openai/gpt-5-nano")


def decompose(question):
    return client.create(
        model="gpt-4o",
        response_model=Iterable[Subquestion],
        messages=[
            {
                "role": "user",
                "content": f"Break this question down into subquestions to solve sequentially: {question}",
            }
        ],
    )


def solve(question, solved_questions, original_question):
    return client.create(
        model="gpt-4o",
        response_model=Answer,
        messages=[
            {
                "role": "user",
                "content": f"""
                    <original_question>
                    {original_question}
                    </original_question>

                    <solved_subquestions>
                    {solved_questions}
                    </solved_subquestions>

                    Solve this next subquestion: {question}
                    """,
            }
        ],
    ).answer


if __name__ == "__main__":
    question = "Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody?"

    # Stage 1: Decompose Question into Subquestions
    subquestions = decompose(question)

    # Stage 2: Sequentially Solve Subquestions
    solved_questions = []
    for subquestion in subquestions:
        solved_questions.append(
            SubquestionWithAnswers(
                question=subquestion.question,
                answer=solve(subquestion, solved_questions, question),
            )
        )

    # Print
    for item in solved_questions:
        print(f"{item.question} {item.answer}")
        #> How old is Mohamed currently? 60
        #> How old was Mohamed four years ago? 56
        #> How old was Kody four years ago if he was half as old as Mohamed? 28
        #> How old is Kody currently? 32
```

### References

<sup id="ref-1">1</sup>: [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)