---
description: "Re2 (Re-Reading) is a technique that asks the model to read the question again."
---

How can we enhance a model's understanding of a query?

Re2 (**Re** - **R** eading) is a technique that asks the model to read the question again.

!!! example "Re-Reading Prompting"
    **Prompt Template**: Read the question again: <*query*> <*critical thinking prompt*><sup><a href="https://arxiv.org/abs/2309.06275">1</a></sup>

    A common critical thinking prompt is: "Let's think step by step."

## Implementation

```python hl_lines="20"
import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-5-nano")


class Response(BaseModel):
    answer: int


def re2(query, thinking_prompt):
    return client.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "system",
                "content": f"Read the question again: {query} {thinking_prompt}",
            },
        ],
    )


if __name__ == "__main__":
    query = """Roger has 5 tennis balls.
        He buys 2 more cans of tennis balls.
        Each can has 3 tennis balls.
        How many tennis balls does he have now?
        """
    thinking_prompt = "Let's think step by step."

    response = re2(query=query, thinking_prompt=thinking_prompt)
    print(response.answer)
    #> 11
```

## References

<sup id="ref-1">1</sup>: [Re-Reading Improves Reasoning in Large Language Models](https://arxiv.org/abs/2309.06275)
