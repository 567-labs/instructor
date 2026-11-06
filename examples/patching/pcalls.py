from typing import Literal, Union
from collections.abc import Iterable
from pydantic import BaseModel
from instructor import ResponseSchema

import time
import openai
import instructor


client = openai.OpenAI()


class Weather(ResponseSchema):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(ResponseSchema):
    query: str


if __name__ == "__main__":

    class Query(BaseModel):
        query: list[Union[Weather, GoogleSearch]]

    client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

    start = time.perf_counter()
    resp = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
    )
    print(f"# Time: {time.perf_counter() - start:.2f}")

    print("# Instructor: Question with Toronto and Super Bowl")
    print([model for model in resp])

    start = time.perf_counter()
    resp = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas?",
            },
        ],
        tools=[
            {"type": "function", "function": Weather.openai_schema},
            {"type": "function", "function": GoogleSearch.openai_schema},
        ],
        tool_choice="auto",
    )
    print(f"# Time: {time.perf_counter() - start:.2f}")

    print("# Question with Toronto and Dallas")
    for tool_call in resp.choices[0].message.tool_calls:
        print(tool_call.model_dump_json(indent=2))

    start = time.perf_counter()
    resp = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in toronto? and who won the super bowl?",
            },
        ],
        tools=[
            {"type": "function", "function": Weather.openai_schema},
            {"type": "function", "function": GoogleSearch.openai_schema},
        ],
        tool_choice="auto",
    )
    print(f"# Time: {time.perf_counter() - start:.2f}")

    print("# Question with Toronto and Super Bowl")
    for tool_call in resp.choices[0].message.tool_calls:
        print(tool_call.model_dump_json(indent=2))
