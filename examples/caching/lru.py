import instructor
from openai import OpenAI
from pydantic import BaseModel
import functools

client = instructor.from_openai(OpenAI())


class UserDetail(BaseModel):
    name: str
    age: int


@functools.lru_cache
def extract(data):
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )


def test_extract():
    import time

    start = time.perf_counter()
    model = extract("Extract jason is 25 years old")
    assert model.name.lower() == "jason"
    assert model.age == 25
    print(f"Time taken: {time.perf_counter() - start}")

    start = time.perf_counter()
    model = extract("Extract jason is 25 years old")
    assert model.name.lower() == "jason"
    assert model.age == 25
    print(f"Time taken: {time.perf_counter() - start}")


if __name__ == "__main__":
    test_extract()
    # Time taken: 0.9267581660533324
    # Time taken: 1.2080417945981026e-06
