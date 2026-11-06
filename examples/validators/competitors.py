from typing import Annotated
from pydantic import BaseModel, ValidationError, AfterValidator
from openai import OpenAI

import instructor

client = instructor.from_openai(OpenAI())


def no_competitors(v: str) -> str:
    # does not allow the competitors of mcdonalds
    competitors = ["burger king", "wendy's", "carl's jr", "jack in the box"]
    for competitor in competitors:
        if competitor in v.lower():
            raise ValueError(
                f"""Let them know that you are work for and are only allowed to talk about mcdonalds.
                Do not apologize. Do not even mention `{competitor}` since they are a a competitor of McDonalds"""
            )
    return v


class Response(BaseModel):
    message: Annotated[str, AfterValidator(no_competitors)]


try:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Response,
        max_retries=2,
        messages=[
            {
                "role": "user",
                "content": "What is your favourite order at burger king?",
            },
        ],
    )  # type: ignore
    print(resp.model_dump_json(indent=2))
except ValidationError as e:
    print(e)
