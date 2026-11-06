import os

from pydantic import BaseModel

from instructor import Mode, from_mistral

try:
    from mistralai.client import Mistral
except ImportError:
    from mistralai import Mistral


class UserDetails(BaseModel):
    name: str
    age: int


# enables `response_model` in chat call
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
instructor_client = from_mistral(
    client=client,
    model="mistral-large-latest",
    mode=Mode.TOOLS,
    max_tokens=1000,
)

resp = instructor_client.create(
    response_model=UserDetails,
    messages=[{"role": "user", "content": "Jason is 10"}],
    temperature=0,
)

print(resp)
