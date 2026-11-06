---
description: "Meta Chain Of Thought involves decomposing an initial query into multiple sub questions. We then aggregate the response from each of these chains as context before prompting another LLM to generate a response"
---

Meta Chain Of Thought (Meta COT) <sup><a href="https://arxiv.org/pdf/2304.13007">1</a></sup>. involves the use of multiple reasoning chains to generate a response to a given query. This helps our model evaluate multiple potential reasoning paths and from there, determine a more accurate answer.

We can implement this using `instructor` as seen below.

```python hl_lines="41-42 57-61 96-99"
import instructor
from pydantic import BaseModel, Field
import asyncio
from typing import Optional
client = instructor.from_provider("openai/gpt-5-nano", async_client=True)


class ReasoningAndResponse(BaseModel):
    intermediate_reasoning: str = Field(
        description="""
    Intermediate reasoning steps"""
    )
    correct_answer: str


class MaybeResponse(BaseModel):
    result: Optional[ReasoningAndResponse]
    error: Optional[bool]
    error_message: Optional[str] = Field(
        description="""Informative explanation of why
        the reasoning chain was unable to generate
        a result"""
    )


class QueryDecomposition(BaseModel):
    queries: list[str] = Field(
        description="""A list of queries that need to be
        answered in order to derive the final answer"""
    )


async def generate_queries(query: str):
    return await client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant that
                decomposes a query into multiple sub-queries.""",
            },
            {"role": "user", "content": query},
        ],
        response_model=QueryDecomposition,
    )


async def generate_reasoning_chain(query: str) -> MaybeResponse:
    return await client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                Given a question and a context,
                answer the question step-by-step.

                Indicate the intermediate reasoning
                steps.
                """,
            },
            {"role": "user", "content": query},
        ],
        response_model=MaybeResponse,
    )


async def batch_reasoning_chains(
    queries: list[str],
) -> list[MaybeResponse]:
    coros = [generate_reasoning_chain(query) for query in queries]
    results = await asyncio.gather(*coros)
    return results


async def generate_response(query: str, context: list[MaybeResponse]):
    formatted_context = "\n".join(
        [
            f"""
            {item.result.intermediate_reasoning}
            {item.result.correct_answer}
            """
            for item in context
            if not item.error and item.result
        ]
    )

    return await client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                Given a question and a context,
                answer the question step-by-step.

                If you are unsure, answer Unknown.
                """,
            },
            {
                "role": "user",
                "content": f"""
                    <question>
                    {query}
                    </question>
                    <context>
                    {formatted_context}
                    </context>
                    """,
            },
        ],
        response_model=ReasoningAndResponse,
    )


if __name__ == "__main__":
    query = """Would Arnold Schwarzenegger have been
    able to deadlift an adult Black rhinoceros at his
    peak strength?"""
    decomposed_queries = asyncio.run(generate_queries(query))

    for generated_query in decomposed_queries.queries:
        print(generated_query)
        #> How much weight could Arnold Schwarzenegger
        #> deadlift at his peak strength?
        #> What is the average weight of an adult Black
        #> rhinoceros?

    chains = asyncio.run(batch_reasoning_chains(decomposed_queries.queries))

    for chain in chains:
        print(chain.model_dump_json(indent=2))
        """
        {
          "result": {
            "intermediate_reasoning": "Determining Arnold
            Schwarzenegger's peak deadlift involves
            researching historical records, interviews,
            and Arnold’s competitive powerlifting
            results.",
            "correct_answer": "Arnold Schwarzenegger's
            peak deadlift was reportedly 710 lbs (322
            kg)."
          },
          "error": false,
          "error_message": null
        }
        """
        """
        {
          "result": {
            "intermediate_reasoning": "To determine the
            average weight of an adult Black rhinoceros,
            I need to consult reliable sources such as
            wildlife encyclopedias, zoological databases,
            or scientific articles. Commonly, the average
            weight of adult Black rhinoceros ranges
            between 800 to 1,400 kg.",
            "correct_answer": "The average weight of an
            adult Black rhinoceros ranges between 800 to
            1,400 kg."
          },
          "error": false,
          "error_message": null
        }
        """

    response = asyncio.run(generate_response(query, chains))

    print(response.model_dump_json(indent=2))
    """
    {
      "intermediate_reasoning": "Arnold Schwarzenegger's
      peak deadlift was 710 lbs (322 kg). The average
      weight of an adult Black rhinoceros ranges between
      800 to 1,400 kg (1764 to 3086 lbs). Even at the
      lower end of the rhinoceros weight range (800 kg
      or 1764 lbs), it exceeds Arnold Schwarzenegger's
      peak deadlift capacity of 710 lbs (322 kg).
      Therefore, Arnold Schwarzenegger would not have
      been able to deadlift an adult Black rhinoceros at
      his peak strength.",
      "correct_answer": "No"
    }
    """
```

### References

<sup id="ref-1">1</sup>: [Answering Questions by Meta-Reasoning over Multiple Chains of Thought](https://arxiv.org/pdf/2304.13007)
