from pydantic import BaseModel, field_validator
from openai import OpenAI
import instructor
import tenacity

client = OpenAI()
client = instructor.from_openai(client)


class User(BaseModel):
    name: str
    age: int

    @field_validator("name")
    def name_is_uppercase(cls, v: str):
        assert v.isupper(), "Name must be uppercase"
        return v


resp = client.messages.create(
    model="gpt-3.5-turbo",
    max_tokens=1024,
    max_retries=tenacity.Retrying(
        stop=tenacity.stop_after_attempt(3),
        before=lambda _: print("before:", _),
        after=lambda _: print("after:", _),
    ),
    messages=[
        {
            "role": "user",
            "content": "Extract John is 18 years old.",
        }
    ],
    response_model=User,
)  # type: ignore

assert isinstance(resp, User)
assert resp.name == "JOHN"  # due to validation
assert resp.age == 18
print(resp)

"""
before: <RetryCallState 4421908816: attempt #1; slept for 0.0; last result: none yet>
after: <RetryCallState 4421908816: attempt #1; slept for 0.0; last result: failed (ValidationError 1 validation error for User
name
  Assertion failed, Name must be uppercase [type=assertion_error, input_value='John', input_type=str]
    For further information visit https://errors.pydantic.dev/2.6/v/assertion_error)>
before: <RetryCallState 4421908816: attempt #2; slept for 0.0; last result: none yet>

name='JOHN' age=18
"""
