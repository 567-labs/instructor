import instructor
import openai
import logging

from pydantic import BaseModel


# Set logging to DEBUG
logging.basicConfig(level=logging.INFO)
logging.getLogger("instructor").setLevel(logging.DEBUG)

client = instructor.from_openai(openai.OpenAI())


class UserDetail(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ],
)  # type: ignore
