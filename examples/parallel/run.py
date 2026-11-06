from __future__ import annotations

import openai
import instructor

from typing import Literal
from collections.abc import Iterable
from pydantic import BaseModel


class Weather(BaseModel):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    query: str


client = openai.OpenAI()

client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

resp = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": "You must always use tools"},
        {
            "role": "user",
            "content": "What is the weather in toronto and dallas and who won the super bowl?",
        },
    ],
    response_model=Iterable[Weather | GoogleSearch],
)

for r in resp:
    print(r)
