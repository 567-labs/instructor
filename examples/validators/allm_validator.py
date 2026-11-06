import asyncio
from typing import Annotated
from pydantic import BaseModel, BeforeValidator
from instructor import llm_validator, patch
from openai import AsyncOpenAI

aclient = AsyncOpenAI()

patch()


class QuestionAnswerNoEvil(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(
            llm_validator("don't say objectionable things", allow_override=True)
        ),
    ]


async def main():
    context = "The according to the devil is to live a life of sin and debauchery."
    question = "What is the meaning of life?"

    try:
        qa: QuestionAnswerNoEvil = await aclient.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=QuestionAnswerNoEvil,
            max_retries=2,
            messages=[
                {
                    "role": "system",
                    "content": "You are a system that answers questions based on the context. Answer exactly what the question asks using the context.",
                },
                {
                    "role": "user",
                    "content": f"using the context: {context}\n\nAnswer the following question: {question}",
                },
            ],
        )  # type: ignore
        print(qa)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    asyncio.run(main())
