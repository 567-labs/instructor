---
description: "Complexity Based Prompting involves choosing examples based on their reasoning steps. If reasoning length isn't available, then we can use proxies such as response length"
---

We can improve the performance of our language models by choosing more complex examples. This refers to examples that have either more reasoning steps or a longer response ( when reasoning steps are not available ).

In the event that no examples are available, we can sample multiple responses and generate an answer based off the top few most complex examples. We can determine the complexity based on the length of their reasoning step in a process known as Complexity Based Consistency
<sup><a href="https://arxiv.org/pdf/2210.00720">1</a></sup> .

We can implement Complexity Based Consistency using `instructor` as seen below.

```python hl_lines="40-42"
import instructor
from pydantic import BaseModel, Field
from textwrap import dedent
import asyncio
from collections import Counter
import random

client = instructor.from_provider("openai/gpt-5-nano", async_client=True)


class ReasoningStep(BaseModel):
    step: int = Field(..., description="The step number")
    subquestion: str = Field(..., description="Subquestion to solve")
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


async def generate_single_response(query: str, context: str) -> Response:
    return await client.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "system",
                "content": dedent(
                    f"""
                You are an expert Question Answering system. Make sure
                to output your reasoning in structured reasoning steps
                before generating a response to the user's query.


                Context:
                {context}

                Query:
                {query}
                """
                ),
            },
        ],
    )


async def complexity_based_consistency(
    query: str, context: str, samples: int, top_k: int
):
    generated_responses = [
        generate_single_response(query, context) for _ in range(samples)
    ]
    responses = await asyncio.gather(*generated_responses)
    sorted_responses = sorted(responses, key=lambda x: len(x.reasoning), reverse=True)
    top_responses = sorted_responses[:top_k]
    return top_responses


if __name__ == "__main__":
    query = "How many loaves of bread did they have left?"
    context = """
    The bakers at the Beverly Hills Bakery baked
    200 loaves of bread on Monday morning. They
    sold 93 loaves in the morning and 39 loaves
    in the afternoon. A grocery store returned 6
    unsold loaves.
    """

    number_of_reasoning_chains = 5
    top_k_to_sample = 3
    response = asyncio.run(
        complexity_based_consistency(
            query, context, number_of_reasoning_chains, top_k_to_sample
        )
    )

    answer_counts = Counter([res.correct_answer for res in response])

    most_common_count = answer_counts.most_common(len(answer_counts))[0][1]
    max_answers = [
        answer for answer, count in answer_counts.items() if count == most_common_count
    ]

    final_answer = random.choice(max_answers)
    print(final_answer)
    #> 74
```

### References

<sup id="ref-1">1</sup>: [Complexity-based prompting for multi-step reasoning](https://arxiv.org/pdf/2210.00720)
