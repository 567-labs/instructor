import time

from collections.abc import Iterable
from openai import OpenAI
from pydantic import BaseModel

import instructor


client = instructor.from_openai(OpenAI())


class User(BaseModel):
    name: str
    job: str
    age: int


def stream_extract(input: str) -> Iterable[User]:
    return client.chat.completions.create_iterable(
        model="gpt-4o",
        temperature=0.1,
        stream=True,
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a perfect entity extraction system",
            },
            {
                "role": "user",
                "content": (
                    f"Consider the data below:\n{input}"
                    "Correctly segment it into entitites"
                    "Make sure the JSON is correct"
                ),
            },
        ],
        max_tokens=1000,
    )


start = time.time()
for user in stream_extract(
    input="Create 5 characters from the book Three Body Problem"
):
    delay = round(time.time() - start, 1)
    print(f"{delay} s: User({user})")
    """
    0.8 s: User(name='Ye Wenjie' job='Astrophysicist' age=60)
    1.1 s: User(name='Wang Miao' job='Nanomaterials Researcher' age=40)
    1.7 s: User(name='Shi Qiang' job='Detective' age=50)
    1.9 s: User(name='Ding Yi' job='Theoretical Physicist' age=45)
    1.9 s: User(name='Chang Weisi' job='Military Strategist' age=55)
    """
    # Notice that the first one would return at 5s bu the last one returned in 10s!
