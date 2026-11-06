import os
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from instructor import Maybe, Mode

# Extract API key from environment
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
assert openrouter_api_key, "OPENROUTER_API_KEY is not set in environment variables"

# Base URL for OpenAI client
openrouter_base_url = os.environ.get("OPENROUTER_BASE_URL")
assert openrouter_base_url, "OPENROUTER_BASE_URL is not set in environment variables"

# Initialize OpenAI client
client = instructor.from_openai(
    OpenAI(api_key=openrouter_api_key, base_url=openrouter_base_url),
    mode=Mode.JSON,
)

data = [
    "Brandon is 33 years old. He works as a solution architect.",
    "Jason is 25 years old. He is the GOAT.",
    "Dominic is 45 years old. He is retired.",
    "Jenny is 72. She is a wife and a CEO.",
    "Holly is 22. She is an explorer.",
    "There onces was a prince, named Benny. He ruled for 10 years, which just ended. He started at 22.",
    "Simon says, why are you 22 years old marvin?",
]


if __name__ == "__main__":

    class UserDetail(BaseModel):
        name: str = Field(description="Name extracted from the text")
        age: int = Field(description="Age extracted from the text")
        occupation: Optional[str] = Field(
            default=None, description="Occupation extracted from the text"
        )

    for content in data:
        MaybeUser = Maybe(UserDetail)
        user = client.chat.completions.create(
            response_model=MaybeUser,
            model="teknium/openhermes-2.5-mistral-7b",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert at outputting json. You always output valid json based on this schema: {MaybeUser.model_json_schema()}",
                },
                {
                    "role": "user",
                    "content": f"Extract the user details from the following text: {content}. Match your response the correct schema",
                },
            ],
        )
        # Output the error or the result.
        if user.error:
            print(f"Error: {user.error}")
        if user.result:
            print(f"Result: {user.result}")
