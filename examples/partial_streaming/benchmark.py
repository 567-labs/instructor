# Adapted from https://cookbook.openai.com/examples/how_to_stream_completions
import time

import instructor
from instructor.v2.providers.openai.schema import make_strict_schema
from openai import OpenAI
from pydantic import BaseModel

MODEL = "gpt-6-luna"
MESSAGES = [
    {
        "role": "user",
        "content": "Describe a Harry Potter character with their name, role, and age.",
    }
]
openai_client = OpenAI()
client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON_SCHEMA)


class User(BaseModel):
    name: str
    role: str
    age: int


def benchmark_raw_stream() -> float:
    # Use the same structured output schema as create_partial.
    response_model = instructor.Partial[User]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": make_strict_schema(response_model.model_json_schema()),
            "strict": True,
        },
    }
    start = time.perf_counter()
    with openai_client.chat.completions.create(
        model=MODEL,
        reasoning_effort="none",
        messages=MESSAGES,
        response_format=response_format,
        stream=True,
    ) as stream:
        output = "".join(
            chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices
        )
    result = User.model_validate_json(output)
    return len(result.model_dump_json()) / (time.perf_counter() - start)


def benchmark_partial_streaming() -> float:
    start = time.perf_counter()
    result = None
    for snapshot in client.create_partial(
        model=MODEL,
        reasoning_effort="none",
        response_model=User,
        messages=MESSAGES,
    ):
        result = snapshot
    if result is None:
        raise RuntimeError("The stream produced no snapshots")
    result = User.model_validate(result.model_dump())
    return len(result.model_dump_json()) / (time.perf_counter() - start)


if __name__ == "__main__":
    raw_rates = [benchmark_raw_stream() for _ in range(10)]
    partial_rates = [benchmark_partial_streaming() for _ in range(10)]
    avg_raw_rate = sum(raw_rates) / len(raw_rates)
    avg_partial_rate = sum(partial_rates) / len(partial_rates)
    print(f"Raw streaming: {avg_raw_rate:.2f} JSON characters/sec")
    print(f"Partial streaming: {avg_partial_rate:.2f} JSON characters/sec")
    print(f"Partial throughput: {avg_partial_rate / avg_raw_rate:.2f}x raw")

    # Historical GPT-3.5 measurements used tokens/sec. Before the earlier
    # optimization, raw and partial streaming measured 35.73 and 24.42.
    # Afterward they measured 35.77 and 31.58.
