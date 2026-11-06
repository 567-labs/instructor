---
title: "System 2 Attention (S2A)"
description: "The S2A (System 2 Attention) technique auto-refines a prompt by asking the model to rewrite the prompt to include only relevant information."
---

How do we remove irrelevant information from the prompt?

The S2A (System 2 Attention) technique auto-refines a prompt by asking the model to rewrite the prompt to include only *relevant* information. We implement this in two steps:

1. Ask the model to rewrite the prompt
2. Pass the rewritten prompt back to the model

## Implementation

```python hl_lines="25-28"
import openai
import instructor
from pydantic import BaseModel, Field

client = instructor.from_provider("openai/gpt-5-nano")


class Step1(BaseModel):
    relevant_context: str = Field(..., description="Relevant context")
    user_query: str = Field(..., description="The question from the user")


class Step2(BaseModel):
    answer: int


def rewrite_prompt(query):
    rewritten_prompt = client.create(
        model="gpt-4o",
        response_model=Step1,
        messages=[
            {
                "role": "user",
                "content": f"""
                    Given the following text by a user, extract the part
                    that is actually relevant to their question. Please
                    include the actual question or query that the user
                    is asking.

                    Text by user:
                    {query}
                    """,  # (1)!
            }
        ],
    )
    return rewritten_prompt


def generate_final_response(rewritten_prompt):
    final_response = client.create(
        model="gpt-4o",
        response_model=Step2,
        messages=[
            {
                "role": "user",
                "content": f"""{rewritten_prompt.relevant_context}
                    Question: {rewritten_prompt.user_query}""",
            }
        ],
    )
    return final_response


if __name__ == "__main__":
    query = """Mary has 3 times as much candy as Megan.
        Mary then adds 10 more pieces of candy to her collection.
        Max is 5 years older than Mary.
        If Megan has 5 pieces of candy, how many does Mary have in total?
        """

    # Step 1: Rewrite the prompt
    rewritten_prompt = rewrite_prompt(query)
    print(rewritten_prompt.relevant_context)
    """
    Mary has 3 times as much candy as Megan. Mary then adds 10 more pieces of candy to her collection. If Megan has 5 pieces of candy, how many does Mary have in total?
    """
    print(rewritten_prompt.user_query)
    #> how many does Mary have in total?

    # Step 2: Generate the final response
    final_response = generate_final_response(rewritten_prompt)
    print(final_response.answer)
    #> 25
```

1. This prompt template comes from [this](https://arxiv.org/abs/2311.11829) paper.

## References

<sup id="ref-1">1</sup>: [System 2 Attention (is something you might need too)](https://arxiv.org/abs/2311.11829)