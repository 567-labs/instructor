import time
import asyncio

import instructor
from pydantic import BaseModel
from openai import AsyncOpenAI


client = instructor.apatch(AsyncOpenAI())


class Timer:
    def __init__(self, name):
        self.name = name
        self.start = None
        self.end = None

    async def __aenter__(self):
        self.start = time.time()

    async def __aexit__(self, *args, **kwargs):
        self.end = time.time()
        print(f"{self.name} took {(self.end - self.start):.2f} seconds")


class Person(BaseModel):
    name: str
    age: int


async def extract_person(text: str) -> Person:
    return await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text},
        ],
        response_model=Person,
    )


async def main():
    """We'll use this to run the example. and time how long each one takes!

    0. for loop
    1. asyncio.gather
    2. asyncio.as_completed
    """
    dataset = [
        "My name is John and I am 20 years old",
        "My name is Mary and I am 21 years old",
        "My name is Bob and I am 22 years old",
        "My name is Alice and I am 23 years old",
        "My name is Jane and I am 24 years old",
        "My name is Joe and I am 25 years old",
        "My name is Jill and I am 26 years old",
    ]

    """
    This is the simplest way to run multiple async functions in series.
    It will wait for each function to complete before continuing.
    """
    async with Timer("for loop"):
        persons = []
        for text in dataset:
            person = await extract_person(text)
            persons.append(person)
        print("for loop:", persons)

    """
    This is the simplest way to run multiple async functions in parallel.
    It will wait for all of the functions to complete before continuing.
    """
    async with Timer("asyncio.gather"):
        tasks_get_persons = [extract_person(text) for text in dataset]
        all_person = await asyncio.gather(*tasks_get_persons)
        print("asyncio.gather:", all_person)

    """
    This is a bit more complicated, but it allows us to process each
    person as soon as they are ready. This is useful if you have a
    large dataset and want to start processing the results as soon
    as they are ready.
    """
    async with Timer("asyncio.as_completed"):
        all_persons = []
        tasks_get_persons = [extract_person(text) for text in dataset]
        for person in asyncio.as_completed(tasks_get_persons):
            all_persons.append(await person)
        print("asyncio.as_copmleted:", all_persons)

    """
    If we want to rate limit our requests, we can use the
    semaphore to limit the number of concurrent requests.
    """

    # Create a semaphore that will only allow 2 concurrent requests
    sem = asyncio.Semaphore(2)

    async def rate_limited_extract_person(text: str) -> Person:
        async with sem:
            return await extract_person(text)

    async with Timer("asyncio.gather (rate limited)"):
        tasks_get_persons = [rate_limited_extract_person(text) for text in dataset]
        resp = await asyncio.gather(*tasks_get_persons)
        print("asyncio.gather (rate limited):", resp)

    async with Timer("asyncio.as_completed (rate limited)"):
        all_persons = []
        tasks_get_persons = [rate_limited_extract_person(text) for text in dataset]
        for person in asyncio.as_completed(tasks_get_persons):
            all_persons.append(await person)
        print("asyncio.as_completed (rate limited):", all_persons)


if __name__ == "__main__":
    asyncio.run(main())
    """
    for loop took 6.17 seconds

    asyncio.gather took 1.11 seconds
    asyncio.as_completed took 0.87 seconds

    asyncio.gather (rate limited) took 3.04 seconds
    asyncio.as_completed (rate limited) took 3.26 seconds
    """
