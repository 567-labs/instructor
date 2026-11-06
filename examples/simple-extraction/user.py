import instructor

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

client = instructor.from_openai(OpenAI())


class UserDetail(BaseModel):
    age: int
    name: str
    role: Optional[str] = Field(default=None)


def get_user_detail(string) -> UserDetail:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=UserDetail,
        messages=[
            {
                "role": "user",
                "content": f"Get user details for {string}",
            },
        ],
    )  # type: ignore


user = get_user_detail("Jason is 25 years old")
print(user.model_dump_json(indent=2))
"""
{
  "age": 25,
  "name": "Jason",
  "role": null
}
"""

user = get_user_detail("Jason is a 25 years old scientist")
print(user.model_dump_json(indent=2))
"""
{
  "age": 25,
  "name": "Jason",
  "role": "scientist"
}
"""

# ! notice that the string should not contain anything
# ! but a user and age was still extracted ?!
user = get_user_detail("User not found")
print(user.model_dump_json(indent=2))
"""
{
  "age": 25,
  "name": "John Doe",
  "role": "null"
}
"""
