---
description: "Demonstration Ensembling(DENSE) creates multiple few-shot prompts, each containing a distinct subset of examples from the training set. We then use that to generate a final response"
---

We can maximise the use of our examples by prompting our model multiple times, each time using a different subset of examples. We can then take these multiple outputs and aggregate over them to generate a final response. This is known as Demonstration Ensembling ( DENSE ) <sup><a href="https://arxiv.org/pdf/2308.08780">1</a></sup>.

> For simplicity in this example, we simply iterate over the examples and partition them equally to get equally sized clusters. However, depending on your use-case you might also want to consider sampling these using some form of embedding clusering.

We can implement this using `instructor` as seen below.

```python hl_lines="26-41"
import instructor
from pydantic import BaseModel
import asyncio
from collections import Counter
from typing import Literal
from textwrap import dedent

class DemonstrationResponse(BaseModel):
    correct_answer: Literal["Positive", "Negative", "Neutral"]


client = instructor.from_provider("openai/gpt-5-nano", async_client=True)


async def generate_self_consistent_response(prompt: str, examples: list[str]):
    concetenated_examples = "\n".join(examples)
    return await client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": dedent(
                    f"""
                You are an intelligent AI System that excels
                at classifying user queries into three
                possible labels:
                - Positive
                - Negative
                - Neutral

                You are about to be given a user query and
                asked to classify it into one of the three
                categories. Make sure to refer closely to
                the examples provided to you, examining each
                individual example before coming up with the
                final answer.

                Here are the examples:
                {concetenated_examples}
                """
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_model=DemonstrationResponse,
        temperature=0,
    )


async def generate_self_consistent_responses(
    prompt: str, num_responses: int, examples: list[str]
):
    assert (
        len(examples) % num_responses == 0
    ), "The number of examples must be evenly divisible by num_responses"

    # Batch the examples into num_responses batches
    batch_size = len(examples) // num_responses

    coros = [
        generate_self_consistent_response(prompt, examples[i : i + batch_size])
        for i in range(0, len(examples), batch_size)
    ]

    responses = await asyncio.gather(*coros)
    return responses


if __name__ == "__main__":
    user_query = "What is the weather like today?"
    examples = [
        "I love this product! [Positive]",
        "This is the worst service ever. [Negative]",
        "The movie was okay, not great but not terrible. [Neutral]",
        "I'm so happy with my new phone! [Positive]",
        "The food was terrible and the service was slow. [Negative]",
        "It's an average day, nothing special. [Neutral]",
        "Fantastic experience, will come again! [Positive]",
        "I wouldn't recommend this to anyone. [Negative]",
        "The book was neither good nor bad. [Neutral]",
        "Absolutely thrilled with the results! [Positive]",
    ]
    responses = asyncio.run(generate_self_consistent_responses(user_query, 5, examples))
    answer_counts = Counter([response.correct_answer for response in responses])
    most_common_answer, _ = answer_counts.most_common(1)[0]
    print(most_common_answer)
    #> Neutral
```

### References

<sup id="ref-1">1</sup>: [Exploring Demonstration Ensembling for In Context Learning](https://arxiv.org/pdf/2308.08780)
