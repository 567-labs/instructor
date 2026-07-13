import os
import instructor
from pydantic import BaseModel, Field
from typing import Optional
from instructor import Maybe

# Requesty (https://requesty.ai) is an OpenAI-compatible LLM router.
# Set your API key: export REQUESTY_API_KEY="your-api-key"
assert os.environ.get("REQUESTY_API_KEY"), (
    "REQUESTY_API_KEY is not set in environment variables"
)

# Requesty uses provider/model naming, e.g. openai/gpt-4o-mini
client = instructor.from_provider("requesty/openai/gpt-4o-mini")

data = [
    "Brandon is 33 years old. He works as a solution architect.",
    "Jason is 25 years old. He is the GOAT.",
    "Dominic is 45 years old. He is retired.",
    "Jenny is 72. She is a wife and a CEO.",
    "Holly is 22. She is an explorer.",
]


class UserDetail(BaseModel):
    age: int
    name: str
    occupation: Optional[str] = Field(
        default=None, description="The person's occupation, if mentioned"
    )


MaybeUser = Maybe(UserDetail)

for text in data:
    user = client.chat.completions.create(
        response_model=MaybeUser,
        messages=[
            {"role": "user", "content": f"Extract the user details: {text}"},
        ],
    )
    print(user)
