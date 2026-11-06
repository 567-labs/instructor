import instructor

from openai import OpenAI
from pydantic import BaseModel


# By default, the patch function will patch the ChatCompletion.create and ChatCompletion.acreate methods. to support response_model parameter
client = instructor.from_openai(
    OpenAI(),
    mode=instructor.Mode.TOOLS,
)


# Now, we can use the response_model parameter using only a base model
# rather than having to use the ResponseSchema class
class UserExtract(BaseModel):
    name: str
    age: int


user: UserExtract = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)  # type: ignore

print(user)
{
    "name": "Jason",
    "age": 25,
}
