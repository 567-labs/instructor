import instructor

from openai import OpenAI
from instructor import llm_validator
from pydantic import BaseModel, ValidationError, BeforeValidator
from typing import Annotated

# Apply the patch to the OpenAI client
client = instructor.from_openai(OpenAI())


class QuestionAnswer(BaseModel):
    question: str
    answer: str


question = "What is the meaning of life?"
context = "The according to the devil is to live a life of sin and debauchery."

qa: QuestionAnswer = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=QuestionAnswer,
    messages=[
        {
            "role": "system",
            "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
        },
        {
            "role": "user",
            "content": f"using the context: {context}\n\nAnswer the following question: {question}",
        },
    ],
)  # type: ignore

print("Before validation with `llm_validator`")
print(qa.model_dump_json(indent=2), end="\n\n")
"""
Before validation with `llm_validator`
{
    "question": "What is the meaning of life?",
    "answer": "The meaning of life, according to the context, is to live a life of sin and debauchery.",
}
"""


class QuestionAnswerNoEvil(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(
            llm_validator("don't say objectionable things", openai_client=client)
        ),
    ]


try:
    qa = QuestionAnswerNoEvil(
        question="What is the meaning of life?",
        answer="The meaning of life is to be evil and steal",
    )
except ValidationError as e:
    print(e)
"""
1 validation error for QuestionAnswerNoEvil
answer
  Assertion failed, The statement promotes objectionable behavior. [type=assertion_error, input_value='The meaning of life is to be evil and steal', input_type=str]
    For further information visit https://errors.pydantic.dev/2.4/v/assertion_error
"""

try:
    qa: QuestionAnswerNoEvil = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=QuestionAnswerNoEvil,
        messages=[
            {
                "role": "system",
                "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
            },
            {
                "role": "user",
                "content": f"using the context: {context}\n\nAnswer the following question: {question}",
            },
        ],
    )  # type: ignore
except Exception as e:
    print(e, end="\n\n")
    """
    1 validation error for QuestionAnswerNoEvil
    answer
        Assertion failed, The statement promotes sin and debauchery, which is objectionable. [type=assertion_error, input_value='The meaning of life is t... of sin and debauchery.', input_type=str]
        For further information visit https://errors.pydantic.dev/2.3/v/assertion_error
    """

qa: QuestionAnswerNoEvil = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=QuestionAnswerNoEvil,
    max_retries=2,
    messages=[
        {
            "role": "system",
            "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
        },
        {
            "role": "user",
            "content": f"using the context: {context}\n\nAnswer the following question: {question}",
        },
    ],
)  # type: ignore

print("After validation with `llm_validator` with `max_retries=2`")
print(qa.model_dump_json(indent=2), end="\n\n")
"""
After validation with `llm_validator` with `max_retries=2`
{
  "question": "What is the meaning of life?",
  "answer": "The meaning of life is subjective and can vary depending on individual beliefs and philosophies."
}
"""
