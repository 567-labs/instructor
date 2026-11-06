import instructor

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

client = instructor.from_openai(OpenAI())


class UserDetail(BaseModel):
    age: int
    name: str
    role: Optional[str] = Field(default=None)


MaybeUser = instructor.Maybe(UserDetail)


def get_user_detail(string) -> MaybeUser:  # type: ignore
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=MaybeUser,
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
  "user": {
    "age": 25,
    "name": "Jason",
    "role": null
  },
  "error": false,
  "message": null
}
"""

user = get_user_detail("Jason is a 25 years old scientist")
print(user.model_dump_json(indent=2))
"""
{
  "user": {
    "age": 25,
    "name": "Jason",
    "role": "scientist"
    },
  "error": false,
  "message": null
}
"""

# ! notice that the string should not contain anything
# ! but a user and age was still extracted ?!
user = get_user_detail("User not found")
print(user.model_dump_json(indent=2))
"""
{
  "user": null,
  "error": true,
  "message": "User not found"
}
"""

# ! due to the __bool__ method, you can use the MaybeUser object as a boolean

if not user:
    print("Detected error")
"""
Detected error
"""
