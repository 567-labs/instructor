---
title: "Self-Ask"
description: "Self-Ask is a technique which use a single prompt to encourage a model to use the answers to sub-problems to correctly generate the overall solution."
---

Models can sometimes correctly answer sub-problems but incorrectly answer the overall query. This is known as the *compositionality gap*<sup><a href="https://arxiv.org/abs/2210.03350">1</a></sup>.

How can we encourage a model to use the answers to sub-problems to correctly generate the overall solution?

Self-Ask is a technique which use a single prompt to:

 - decide if follow-up questions are required
 - generate the follow-up questions
 - answer the follow-up questions
 - answer the main query

## Implementation

```python hl_lines="26-29"
import instructor
from pydantic import BaseModel, Field
client = instructor.from_provider("openai/gpt-5-nano")


class FollowUp(BaseModel):
    question: str = Field(description="The follow-up question")
    answer: str = Field(description="The answer to the follow-up question")


class Response(BaseModel):
    follow_ups_required: bool
    follow_ups: list[FollowUp]
    final_answer: str


def self_ask(query):
    return client.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "system",
                "content": f"""Query: {query}
                        Are follow-up questions needed?
                        If so, generate follow-up questions, their answers, and then the final answer to the query.
                        """,  # !(1)
            },
        ],
    )


if __name__ == "__main__":
    query = "Who was president of the U.S. when superconductivity was discovered?"

    response = self_ask(query)

    print(response.follow_ups_required)
    #> True
    for follow_up in response.follow_ups:
        print(follow_up)
        """
        question='When was superconductivity discovered?' answer='Superconductivity was discovered in April 1911.'
        """
        """
        question='Who was president of the U.S. in April 1911?' answer='William Howard Taft was the President of the United States in April 1911.'
        """
    print(response.final_answer)
    """
    William Howard Taft was president of the U.S. when superconductivity was discovered.
    """
```

1. Without `instructor`, this prompt would generally be implemented as a one-shot or few-shot prompt<sup><a href="https://arxiv.org/abs/2210.03350">1</a></sup> to encourage thinking through follow-up questions. With `instructor`, we use a zero-shot prompt!

## References

<sup id="ref-1">1</sup>: [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350)
