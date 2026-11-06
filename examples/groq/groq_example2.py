import os
from pydantic import BaseModel
from groq import Groq
import instructor

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)


class UserExtract(BaseModel):
    name: str
    age: int


user: UserExtract = client.chat.completions.create(
    model="mixtral-8x7b-32768",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

assert isinstance(user, UserExtract), "Should be instance of UserExtract"
assert user.name.lower() == "jason"
assert user.age == 25

print(user.model_dump_json(indent=2))
"""
{
  "name": "jason",
  "age": 25
}
"""
