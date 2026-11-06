import instructor

from instructor import openai_moderation

from typing import Annotated
from pydantic import BaseModel, AfterValidator
from openai import OpenAI

client = instructor.from_openai(OpenAI())


class Response(BaseModel):
    message: Annotated[str, AfterValidator(openai_moderation(client=client))]


response = Response(message="I want to make them suffer the consequences")
