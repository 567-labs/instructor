---
description: "Uncertainty Routed Chain Of Thought is a technique used in the Gemini Paper to improve upon the conventional Chain Of Thought approach"
---

Uncertainty-Routed Chain Of Thought<sup><a href="https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf">1</a></sup> prompting generates multiple chain of thought reasoning chains ( This is either 8 or 32 in the original paper ).

It then takes the majority answer out of these chains as the final solution only if the proportion of chains that agreed on this answer are higher than a specific threshold.

We can implement this using `instructor` as seen below.

```python hl_lines="74-87"
from pydantic import BaseModel
import instructor
from textwrap import dedent
from typing import Literal
import asyncio
from collections import Counter
client = instructor.from_provider("openai/gpt-5-nano", async_client=True)


class ChainOfThoughtResponse(BaseModel):
    chain_of_thought: str
    correct_answer: Literal["A", "B", "C", "D"]


async def generate_response(query: str, options: dict[str, str]):
    formatted_options = "\n".join(
        [f"{key}:{answer}" for key, answer in options.items()]
    )
    return await client.create(
        model="gpt-4o",
        response_model=ChainOfThoughtResponse,
        messages=[
            {
                "role": "system",
                "content": dedent(
                    f"""
                You are a world class AI who excels at answering
                complex questions. Choose one of the options below
                that best answers the question you are about to be
                asked
                <question>
                {query}
                </question>

                <options>
                {formatted_options}
                </options>
                """
                ),
            }
        ],
    )


async def generate_batch_responses(
    query: str, options: dict[str, str], num_chains: int
) -> list[ChainOfThoughtResponse]:
    coros = [generate_response(query, options) for _ in range(num_chains)]
    return await asyncio.gather(*coros)


if __name__ == "__main__":
    question = """In a population of giraffes, an environmental
    change occurs that favors individuals that are tallest. As a
    result, more of the taller individuals are able to obtain
    nutrients and survive to pass along their genetic information.
    This is an example of"""

    options = {
        "A": "directional selection",
        "B": "stabilizing selection",
        "C": "sexual selection",
        "D": "disruptive selection",
    }

    correct_answer = "A"
    k = 8
    threshold = 0.6

    responses = asyncio.run(generate_batch_responses(question, options, k))
    votes = Counter([response.correct_answer for response in responses])
    print(votes)
    #> Counter({'A': 8})

    majority_vote_element, majority_vote_count = votes.most_common(1)[0]
    print(majority_vote_element, majority_vote_count)
    #> A 8
    majority_threshold = majority_vote_count / k

    if majority_threshold < threshold:
        response = asyncio.run(generate_response(question, options))
        response = response.correct_answer
    else:
        response = majority_vote_element

    print(response)
    #> A
```

### References

<sup id="ref-1">1</sup>: [Gemini: A Family of Highly Capable Multimodal Models](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)
