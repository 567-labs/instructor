---
description: "Tab-CoT encourages LLMs to output reasoning as a markdown table, improving the structure and reasoning of its output"
---

By getting language models to output their reasoning as a structured markdown table, we can improve their reasoning capabilities and the quality of their outputs. This is known as Tabular Chain Of Thought (Tab-CoT) <sup><a href="https://arxiv.org/pdf/2305.17812">1</a></sup>.

We can implement this using `instructor` as a response object as seen below to ensure we get exactly the data that we want. Each row in our table is represented here as a `ReasoningStep` object.

```python hl_lines="36-38"
import instructor
from pydantic import BaseModel, Field
from textwrap import dedent
client = instructor.from_provider("openai/gpt-5-nano")


class ReasoningStep(BaseModel):
    step: int = Field(description="The step number")
    subquestion: str = Field(description="Subquestion to solve")
    procedure: str = Field(
        description="""Any intermediate computation
        that was done in the reasoning process. Leave
        empty if no computation is needed""",
    )
    result: str


class Response(BaseModel):
    reasoning: list[ReasoningStep] = Field(
        description="reasoning steps to derive answer",
    )
    correct_answer: int


def generate_structured_reasoning_response(query: str, context: str):
    response = client.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "system",
                "content": dedent(
                    f"""
                <system>
                    <role>expert Question Answering system</role>
                    <instruction>Make sure to output your reasoning in structured reasoning steps before generating a response to the user's query.</instruction>
                </system>

                <context>
                    {context}
                </context>

                <query>
                    {query}
                </query>
                """
                ),
            },
        ],
    )
    return response


if __name__ == "__main__":
    query = "How many loaves of bread did they have left?"
    context = """
    The bakers at the Beverly Hills Bakery baked
    200 loaves of bread on Monday morning. They
    sold 93 loaves in the morning and 39 loaves
    in the afternoon. A grocery store returned 6
    unsold loaves.
    """

    response = generate_structured_reasoning_response(query, context)
    print(response.model_dump_json(indent=2))
    """
    {
      "reasoning": [
        {
          "step": 1,
          "subquestion": "How many loaves of bread were sold in the morning
          and afternoon?",
          "procedure": "93 (morning) + 39 (afternoon)",
          "result": "132"
        },
        {
          "step": 2,
          "subquestion": "How many loaves of bread were originally baked?",
          "procedure": "",
          "result": "200"
        },
        {
          "step": 3,
          "subquestion": "How many loaves of bread were returned by the
          grocery store?",
          "procedure": "",
          "result": "6"
        },
        {
          "step": 4,
          "subquestion": "How many loaves of bread were left after accounting
          for sales and returns?",
          "procedure": "200 (originally baked) - 132 (sold) + 6 (returned)",
          "result": "74"
        }
      ],
      "correct_answer": 74
    }
    """
```

This generates the following reasoning step and the correct response of 74.

| Step | Subquestion                                                                | Procedure                                          | Result |
| ---- | -------------------------------------------------------------------------- | -------------------------------------------------- | ------ |
| 1    | How many loaves of bread were sold in the morning and afternoon?           | 93 (morning) + 39 (afternoon)                      | 132    |
| 2    | How many loaves of bread were originally baked?                            |                                                    | 200    |
| 3    | How many loaves of bread were returned by the grocery store?               |                                                    | 6      |
| 4    | How many loaves of bread were left after accounting for sales and returns? | 200 (originally baked) - 132 (sold) + 6 (returned) | 74     |

### References

<sup id="ref-1">1</sup>: [Tab-CoT: Zero-shot Tabular Chain of Thought](https://arxiv.org/pdf/2305.17812)
