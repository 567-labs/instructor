---
description: "Self Consistency aims to help maximise llm performance by sampling multiple potential calls. We then take a majority vote on the final response to derive the answer"
---

By generating multiple candidate responses in parallel and choosing the most common answer among them, we can get a more accurate answer. This is known as Self-Consistency <sup><a href="https://arxiv.org/pdf/2203.11171">1</a></sup>

We can implement this using `instructor` as seen below.

```python hl_lines="25-29"
import instructor
from pydantic import BaseModel, Field
import asyncio
from collections import Counter
from textwrap import dedent

class SelfConsistencyResponse(BaseModel):
    chain_of_thought: str = Field(
        description="reasoning behind the final correct answer"
    )
    correct_answer: int


client = instructor.from_provider("openai/gpt-5-nano", async_client=True)


async def generate_self_consistent_response(prompt: str):
    return await client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an intelligent question
                answering AI system that excels at answering
                user queries. Make sure to generate a
                comprehensive explanation of your thought
                process before providing the final answer""",
            },
            {"role": "user", "content": prompt},
        ],
        response_model=SelfConsistencyResponse,
        temperature=0.5,
    )


async def generate_self_consistent_responses(prompt: str, num_responses: int):
    coros = [generate_self_consistent_response(prompt) for _ in range(num_responses)]
    responses = await asyncio.gather(*coros)
    return responses


if __name__ == "__main__":
    prompt = dedent(
        """
        Janet's ducks lay 16 eggs per day.
        She eats three for breakfast every
        morning and bakes muffins for her
        friends every day with four. She sells
        the remainder for $2 per egg. How
        much does she make every day?
        """
    )
    responses = asyncio.run(generate_self_consistent_responses(prompt, 5))
    answer_counts = Counter([response.correct_answer for response in responses])
    most_common_answer, _ = answer_counts.most_common(1)[0]

    print(most_common_answer)
    #> 18
```

### References

<sup id="ref-1">1</sup>: [Self-Consistency Improves Chain Of Thought
Reasoning In Language Models](https://arxiv.org/pdf/2210.03350)
