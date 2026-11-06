---
description: "We get a model to output a baseline response. Next, we independently verify the response by using a model to generate questions and to verify these questions. Lastly, we use a final API call to verify the baseline response with the generated data"
---

Chain Of Verification ( CoVe )<sup><a href="https://arxiv.org/pdf/2309.11495">1</a></sup> is a method that allows us to be able to verify our LLM's generated responses. We can do so using the following steps

1. First we get our LLM to generate a response to a query
2. Then we generate a set of follow up questions that need to be answered to validate the response
3. We then independently generate a set of responses to these questions
4. Lastly, we use a final LLM call to verify the response in light of these new question and answer pairs that we've generated

```python hl_lines="49-52 95-100"
import instructor
from pydantic import BaseModel, Field
import asyncio
client = instructor.from_provider("openai/gpt-5-nano", async_client=True)


class QueryResponse(BaseModel):
    correct_answer: str


class ValidationQuestions(BaseModel):
    question: list[str] = Field(
        description="""A list of questions that need to be
        answered to validate the response"""
    )


class ValidationAnswer(BaseModel):
    answer: str


class FinalResponse(BaseModel):
    correct_answer: str


async def generate_initial_response(query: str):
    return await client.create(
        model="gpt-4o",
        response_model=QueryResponse,
        messages=[
            {
                "role": "system",
                "content": "You are an expert question answering system",
            },
            {"role": "user", "content": query},
        ],
    )


async def generate_verification_questions(llm_response: str):
    return await client.create(
        model="gpt-4o",
        response_model=ValidationQuestions,
        messages=[
            {
                "role": "system",
                "content": """You are an expert AI system that excels at
                generating follow up questions to validate a response.
                These questions should validate key assumptions, facts
                and other important portions of the generated response""",
            },
            {"role": "user", "content": llm_response},
        ],
    )


async def generate_verification_response(questions: list[str]):
    async def verify_question(question: str) -> tuple[ValidationAnswer, str]:
        return (
            await client.create(
                model="gpt-4o",
                response_model=ValidationAnswer,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert AI system that
                        excels at answering validation questions.""",
                    },
                    {"role": "user", "content": question},
                ],
            ),
            question,
        )

    coros = [verify_question(question) for question in questions]
    return await asyncio.gather(*coros)


async def generate_final_response(
    answers: list[tuple[ValidationAnswer, str]],
    initial_response: QueryResponse,
    original_query: str,
):
    formatted_answers = "\n".join(
        [f"Q: {question}\nA: {answer.answer}" for answer, question in answers]
    )
    return await client.create(
        model="gpt-4o",
        response_model=FinalResponse,
        messages=[
            {
                "role": "system",
                "content": """You are an expert AI system that excels at
                validating and verifying if an initial answer answers an
                initial query based off some Verification Questions and
                Answers provided. Return the original answer if it is
                valid else generate a new response off the verification
                questions and answers provided.""",
            },
            {
                "role": "user",
                "content": f"""
                Initial query: {original_query}
                Initial Answer : {initial_response.correct_answer}
                Verification Questions and Answers:
                {formatted_answers}
            """,
            },
        ],
    )


if __name__ == "__main__":
    query = "What was the primary cause of the Mexican-American war and how long did it last?"
    initial_response = asyncio.run(generate_initial_response(query))
    print(initial_response.model_dump_json())
    """
    {"correct_answer":"The primary cause of the Mexican-American War was
    the annexation of Texas by the United States and the dispute over
    whether Texas ended at the Nueces River (as the Mexicans claimed) or
    the Rio Grande (as the U.S. claimed). The war lasted from April 25,
    1846, to February 2, 1848, totaling nearly two years."}
    """

    verification_questions = asyncio.run(
        generate_verification_questions(initial_response.correct_answer)
    )
    print(verification_questions.model_dump_json())
    """
    {"question":["Is it accurate that the primary cause of the
    Mexican-American War was the annexation of Texas by the United
    States?","Was there a dispute over whether Texas ended at the Nueces
    River or the Rio Grande?","Did the Mexican-American War last from
    April 25, 1846, to February 2, 1848?","Is it correct to state that
    the disagreement over the Texas border was between the Nueces River
    and the Rio Grande?","Was the Mexican claim that Texas ended at the
    Nueces River while the U.S. claimed it was at the Rio Grande?"]}
    """

    responses = asyncio.run(
        generate_verification_response(verification_questions.question)
    )

    final_answer = asyncio.run(
        generate_final_response(responses, initial_response, query)
    )
    print(final_answer.model_dump_json())
    """
    {"correct_answer":"The primary cause of the Mexican-American War was
    the annexation of Texas by the United States and the dispute over
    whether Texas ended at the Nueces River (as the Mexicans claimed) or
    the Rio Grande (as the U.S. claimed). The war lasted from April 25,
    1846, to February 2, 1848, totaling nearly two years."}
    """
```

### References

<sup id="ref-1">1</sup>: [Chain-Of-Verification Reduces Hallucination In Large Language Models](https://arxiv.org/pdf/2309.11495)
