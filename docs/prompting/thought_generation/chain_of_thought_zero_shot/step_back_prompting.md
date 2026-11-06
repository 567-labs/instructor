---
description: "Step-back prompting is a two-step prompting technique that asks the LLM a step-back question to gather context for the query"
---

How can we encourage an LLM to think through any high-level context required to answer a query? Step-back prompting encourages this in two steps:

1. **Abstraction**: Ask the LLM a generic, higher-level concept. This is generally topic-specific. This is known as the _step-back question_.
2. **Reasoning**: Ask the LLM the original question, given its answer to the abstract question. This is known as _abstracted-grounded reasoning_.

!!! example "Step-Back Prompting Example"

    **Original Question**: What happens to the pressure of an ideal gas when temperature and volume are increased?

    **Step-Back Question**: What are the physics concepts associated with this question?

    **Reasoning Prompt**: {step-back response} {original question}

Note that the step-back question is also generated using an LLM query.

Step-back prompting has been shown to improve scores on reasoning benchmarks for PaLM-2L and GPT-4.<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>

```python
import openai
import instructor
from pydantic import BaseModel
from typing import Iterable, Literal

client = instructor.from_provider("openai/gpt-5-nano")


class Stepback(BaseModel):
    original_question: str
    abstract_question: str


class Education(BaseModel):
    degree: Literal["Bachelors", "Masters", "PhD"]
    school: str
    topic: str
    year: int


class Response(BaseModel):
    school: str


def generate_stepback_question():
    return client.create(
        model="gpt-4o",
        response_model=Stepback,
        messages=[
            {
                "role": "user",
                "content": f"""
                You are an expert at world knowledge. Your task is to step back
                and paraphrase a question to a more generic step-back question,
                which is easier to answer.

                Here are a few examples:
                Original Question: Which position did Knox Cunningham hold from
                May 1955 to Apr 1956?
                Step-back Question: Which positions has Knox Cunningham held in
                his career?
                Original Question: Who was the spouse of Anna Karina from 1968
                to 1974?
                Step-back Question: Who were the spouses of Anna Karina?
                Original Question: Which team did Thierry Audel play for from
                2007 to 2008?
                Step-back Question: Which teams did Thierry Audel play for in
                his career?

                Now, generate the step-back question for the following question:
                Estella Leopold went to which school between Aug 1954 and
                Nov 1954?
                """,
            },
        ],
    )


def ask_stepback_question(stepback):
    return client.create(
        model="gpt-4o",
        response_model=Iterable[Education],
        messages=[
            {"role": "user", "content": stepback.abstract_question},
        ],
    )


def get_final_response(stepback, stepback_response):
    return client.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "user",
                "content": f"""
                Q: {stepback.abstract_question},
                A: {stepback_response}
                Q: {stepback.original_question}
                A:
                """,
            },
        ],
    )


if __name__ == "__main__":
    # Generate the step-back question
    stepback = generate_stepback_question()
    print(stepback.original_question)
    #> Estella Leopold went to which school between Aug 1954 and Nov 1954?
    print(stepback.abstract_question)
    #> Which schools did Estella Leopold attend in her life?

    # Ask the step-back question
    stepback_response = ask_stepback_question(stepback)
    for item in stepback_response:
        print(item)
        """
        degree='Bachelors'
        school='University of Wisconsin-Madison'
        topic='Botany'
        year=1948
        """
        """
        degree='Masters'
        school='University of California, Berkeley'
        topic='Botany and Paleobotany'
        year=1950
        """
        """
        degree='PhD'
        school='Yale University'
        topic='Botany and Paleobotany'
        year=1955
        """

    # Ask the original question, appended with context from the stepback response
    print(get_final_response(stepback, stepback_response))
    #> school='Yale University'
```

### References

<sup id="ref-1">1</sup>: [Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://arxiv.org/abs/2310.06117)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
