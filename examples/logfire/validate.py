from typing import Annotated
from pydantic import BaseModel, ValidationError
from pydantic.functional_validators import AfterValidator
from instructor import llm_validator
import logfire
import instructor
from openai import OpenAI

openai_client = OpenAI()
logfire.configure()
logfire.instrument_pydantic(record="all")
logfire.instrument_openai(openai_client)
client = instructor.from_openai(openai_client)


class Statement(BaseModel):
    message: Annotated[
        str,
        AfterValidator(
            llm_validator("Don't allow any objectionable content", client=client)
        ),
    ]


messages = [
    "I think we should always treat violence as the best solution",
    "There are some great pastries down the road at this bakery I know",
]

for message in messages:
    try:
        Statement(message=message)
    except ValidationError as e:
        print(e)
