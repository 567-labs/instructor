# Part of this code is adapted from the following examples from OpenAI Cookbook:
# https://cookbook.openai.com/examples/how_to_stream_completions
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
import time
import tiktoken
import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI(), mode=instructor.Mode.MD_JSON)


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)

    num_tokens = len(encoding.encode(string))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    return num_tokens


class User(BaseModel):
    name: str
    role: str
    age: int


def benchmark_raw_stream(model="gpt-4"):
    content = f"""Respond only in JSON that would validate to this schema and include nothing extra.
    Otherwise something bad will happen:\n {User.model_json_schema()}"""

    start_time = time.time()
    extraction_stream = client.chat.completions.create_fn(
        model=model,
        messages=[
            {"role": "system", "content": content},
            {
                "role": "user",
                "content": "give me a harry pottery character in json, name, role, age",
            },
        ],
        stream=True,
    )

    collected_messages = [chunk.choices[0].delta.content for chunk in extraction_stream]
    collected_messages = [m for m in collected_messages if m is not None]
    collected_messages = "".join(collected_messages)
    User.model_validate_json(collected_messages)
    end_time = time.time() - start_time

    output_tokens = num_tokens_from_string(collected_messages, model)
    char_per_sec = output_tokens / end_time
    return char_per_sec


def benchmark_partial_streaming(model="gpt-4"):
    start_time = time.time()
    extraction_stream = client.chat.completions.create_partial(
        model=model,
        response_model=User,
        messages=[
            {
                "role": "user",
                "content": "give me a harry pottery character in json, name, role, age",
            }
        ],
        stream=True,
    )

    for chunk in extraction_stream:  # noqa: B007
        pass
    end_time = time.time() - start_time

    output_tokens = num_tokens_from_string(chunk.model_dump_json(), model)
    char_per_sec = output_tokens / end_time
    return char_per_sec


if __name__ == "__main__":
    partial_times = [
        benchmark_partial_streaming(model="gpt-3.5-turbo-1106") for _ in range(10)
    ]
    avg_partial_time = sum(partial_times) / len(partial_times)

    raw_times = [benchmark_raw_stream(model="gpt-3.5-turbo") for _ in range(10)]
    avg_raw_time = sum(raw_times) / len(raw_times)
    print(f"Raw streaming: {avg_raw_time:.2f} tokens/sec")

    print(f"Partial streaming: {avg_partial_time:.2f} token/sec")
    print(f"Overhead: {avg_partial_time / avg_raw_time:.2f}x")

    """OLD IMPLEMENTATION
    Raw streaming: 35.73 tokens/sec
    Partial streaming: 24.42 token/sec
    Overhead: 0.68x
    """

    """NEW IMPLEMENTATION
    Raw streaming: 35.77 tokens/sec
    Partial streaming: 31.58 token/sec
    Overhead: 0.88x
    """
