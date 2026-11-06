# Part of this code is adapted from the following examples from OpenAI Cookbook:
# https://cookbook.openai.com/examples/how_to_stream_completions
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI(), mode=instructor.Mode.TOOLS)


class User(BaseModel):
    name: str
    role: str


extraction_stream = client.chat.completions.create_partial(
    model="gpt-4",
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "give me a harry pottery character in json, name, role, age",
        }
    ],
)

for chunk in extraction_stream:
    print(chunk)
