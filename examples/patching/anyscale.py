import os
import instructor

from openai import OpenAI
from pydantic import BaseModel


# By default, the patch function will patch the ChatCompletion.create and ChatCompletion.acreate methods. to support response_model parameter
client = instructor.from_openai(
    OpenAI(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key=os.environ["ANYSCALE_API_KEY"],
    ),
    mode=instructor.Mode.JSON_SCHEMA,
)


# Now, we can use the response_model parameter using only a base model
# rather than having to use the ResponseSchema class
class UserExtract(BaseModel):
    name: str
    age: int


user: UserExtract = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
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
