---
description: "To help the model better infer human intention from ambiguous prompts, we can ask the model to rephrase and respond (RaR)."
---

How can we identify and clarify ambiguous information in the prompt?

Let's say we are given the query: *Was Ed Sheeran born on an odd month?*

There are many ways a model might interpret an *odd month*:

- February is *odd* because of an irregular number of days.
- A month is *odd* if it has an odd number of days.
- A month is *odd* if its numerical order in the year is odd (i.e. January is the 1st month).

!!! note

    Ambiguities might not always be so obvious!

To help the model better infer human intention from ambiguous prompts, we can ask the model to rephrase and respond (RaR).

## Implementation

```python hl_lines="19"
from pydantic import BaseModel
import instructor
client = instructor.from_provider("openai/gpt-5-nano")


class Response(BaseModel):
    rephrased_question: str
    answer: str


def rephrase_and_respond(query):
    return client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""{query}\nRephrase and expand the question, and respond.""",  # (1)!
            }
        ],
        response_model=Response,
    )


if __name__ == "__main__":
    query = "Take the last letters of the words in 'Edgar Bob' and concatenate them."

    response = rephrase_and_respond(query)

    print(response.rephrased_question)
    """
    What are the last letters of each word in the name 'Edgar Bob', and what do you get when you concatenate them?
    """
    print(response.answer)
    """
    To find the last letters of each word in the name 'Edgar Bob', we look at 'Edgar' and 'Bob'. The last letter of 'Edgar' is 'r' and the last letter of 'Bob' is 'b'. Concatenating these letters gives us 'rb'.
    """
```

1. This prompt template comes from [this](https://arxiv.org/abs/2311.04205) paper.

This can also be implemented as two-step RaR:

1. Ask the model to rephrase the question.
2. Pass the rephrased question back to the model to generate the final response.

## References

<sup id="ref-1">1</sup>: [Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves](https://arxiv.org/abs/2311.04205)
