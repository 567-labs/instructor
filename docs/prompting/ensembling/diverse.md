---
description: "Diverse creates multiple prompts for a given problem before performing self-consistency for each. It then generates multiple reaosning paths before choosing the best final response"
---

Diverse Verifier On Reasoning Step (DiVeRSe)<sup><a href="https://aclanthology.org/2023.acl-long.291/">1</a></sup> is a prompting technique which provides two main improvements

1. **Diverse Prompts** : They generate multiple variations of the same prompt by varying the examples used in each prompt
2. **Verification** : They use a finetuned `Deberta-V3-Large` to determine the quality of a generated response. Instead of using majority voting, they use their model to score each generated response from 0 to 1. They then aggregate these scores for each unique answer to determine the best generated solution.

In the paper itself, they also train a step-wise verifier that is able to score each individual reasoning step. This enables much more fine-grained predictions but is challenging to obtain training data for.

We can implement this in `instructor`. However, instead of using a `deberta-v3-large` model, we'll be using gpt-4o to score its own outputs and generate a quality score.

```python
import instructor
from pydantic import BaseModel
from typing import Literal
from textwrap import dedent
import asyncio
from collections import defaultdict
client = instructor.from_provider("openai/gpt-5-nano", async_client=True)


class Response(BaseModel):
    chain_of_thought: str
    answer: int


class Grading(BaseModel):
    grade: Literal["Poor", "Average", "Good", "Excellent"]

    def get_score(self):
        mapping = {
            "Poor": 0.25,
            "Average": 0.5,
            "Good": 0.75,
            "Excellent": 1,
        }
        return mapping[self.grade]


async def generate_response(query: str, examples: list[str]):
    formatted_examples = "\n".join(examples)
    return await client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": dedent(
                    f"""
                You are a world class AI that excels at answering
                user queries in a succinct and accurate manner.

                <query>
                {query}
                </query>

                <examples>
                {formatted_examples}
                </examples>
                """
                ),
            }
        ],
        response_model=Response,
    )


async def score_response(query: str, response: Response) -> tuple[Response, Grading]:
    return (
        response,
        await client.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": dedent(
                        f"""
                You are a world class AI that excels at grading
                responses to a user query in a succinct and clear
                manner.

                <query>
                {query}
                </query>

                <response>
                {response}
                </response>
                """
                    ),
                }
            ],
            response_model=Grading,
        ),
    )


async def generate_response_batch(
    query: str, examples: list[str], n_examples_per_batch: int
):
    batches: list[list[str]] = []
    for i in range(0, len(examples), n_examples_per_batch):
        batches.append(examples[i : i + n_examples_per_batch])

    coros = [generate_response(query, example_batch) for example_batch in batches]
    return await asyncio.gather(*coros)


async def score_responses(
    query: str, responses: list[Response]
) -> list[tuple[Response, Grading]]:
    coros = [score_response(query, response) for response in responses]
    return await asyncio.gather(*coros)


if __name__ == "__main__":
    examples = [
        """
        Q: James decides to run 3 sprints 3 times a week.
        He runs 60 meters each sprint. How many total
        meters does he run a week?
        A: James decides to run 3 sprints 3 times a week.
        He runs 60 meters each sprint. So he runs 60 meters
        x 3 sprints x 3 times a week. That is 60 meters x 9.
        The answer is 540.
        """,
        """
        Q: Brandon's iPhone is four times as old as Ben's
        iPhone. Ben's iPhone is two times older than Suzy's
        iPhone. If Suzy's iPhone is 1 year old, how old is
        Brandon's iPhone?
        A: Brandon's iPhone is 4 times as old as Ben's
        iPhone. Ben's iPhone is 2 times older than Suzy's
        iPhone. So Brandon's iPhone is 4 x 2 = 8 times older
        than Suzy's iPhone. Suzy's iPhone is 1 year old. So
        Brandon's iPhone is 8 x 1 = 8 years old. The answer
        is 8.
        """,
        """
        Q: Jean has 30 lollipops. Jean eats 2 of the
        lollipops. With the remaining lollipops, Jean wants
        to package 2 lollipops in one bag. How many bags can
        Jean fill?
        A: Jean started with 30 lollipops. She ate 2 of
        them. So she has 28 lollipops left. She wants to
        package 2 lollipops in one bag. So she can package
        28 / 2 = 14 bags. The answer is 14.
        """,
        """
        Q: Weng earns $12 an hour for babysitting.
        Yesterday, she just did 50 minutes of babysitting.
        How much did she earn?
        A: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
        Working 50 minutes, she earned 0.2 x 50 =
        $<<0.2*50=10>>10. The answer is 10
        """,
    ]

    query = """Betty is saving money for a new wallet which
    costs $100. Betty has only half of the money she needs.
    Her parents decided to give her $15 for that purpose,
    and her grandparents twice as much as her parents. How
    much more money does Betty need to buy the wallet?"""

    generated_responses = asyncio.run(generate_response_batch(query, examples, 1))

    scored_responses = asyncio.run(score_responses(query, generated_responses))

    scores: dict[int, float] = defaultdict(int)

    for response, grade in scored_responses:
        scores[response.answer] += grade.get_score()

    print(scores)
    #> defaultdict(<class 'int'>, {5: 3.5})

    answer = max(scores, key=scores.get)
    print(answer)
    #> 5
```

### References

<sup id="ref-1">1</sup>: [Making Language Models Better Reasoners with Step-Aware Verifier](https://aclanthology.org/2023.acl-long.291/)
