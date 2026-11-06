import os
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from instructor import Maybe, Mode

# Extract API key from environment
perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
assert perplexity_api_key, "PERPLEXITY_API_KEY is not set in environment variables"

# Base URL for OpenAI
perplexity_base_url = os.environ.get("PERPLEXITY_BASE_URL")
assert perplexity_base_url, "PERPLEXITY_BASE_URL is not set in environment variables"

# Initialize OpenAI client
client = instructor.from_openai(
    OpenAI(api_key=perplexity_api_key, base_url=perplexity_base_url),
    mode=Mode.JSON,
)

# For direct reference here. See https://docs.perplexity.ai/docs/model-cards for updates
# Recommended is pplx-70b-chat
models = [
    "codellama-34b-instruct",
    "llama-2-70b-chat",
    "mistral-7b-instruct",
    "pplx-7b-chat",
    "pplx-70b-chat",
    "pplx-7b-online",
    "pplx-70b-online",
]

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
            model="pplx-70b-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at outputting json. You always output valid JSON based on the pydantic schema given to you.",
                },
                {
                    "role": "user",
                    "content": f"Extract the user details from the following text: {content}. Match your response to the following schema: {MaybeUser.model_json_schema()}",
                },
            ],
            max_retries=3,
        )
        # Output the error or the result.
        if user.error:
            print(f"Error: {user.error}")
        if user.result:
            print(f"Result: {user.result}")
