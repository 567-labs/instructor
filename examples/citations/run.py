from typing import Optional
from openai import OpenAI
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

import instructor

client = instructor.from_openai(OpenAI())

""" 
Example 1) Simple Substring check that compares a citation to a text chunk
"""


class Statements(BaseModel):
    body: str
    substring_quote: str

    @field_validator("substring_quote")
    @classmethod
    def substring_quote_exists(cls, v: str, info: ValidationInfo):
        context = info.context.get("text_chunks", None)

        # Check if the substring_quote is in the text_chunk
        # if not, raise an error
        for text_chunk in context.values():
            if v in text_chunk:
                return v
        raise ValueError(
            f"Could not find substring_quote `{v}` in contexts",
        )


class AnswerWithCitaton(BaseModel):
    question: str
    answer: list[Statements]


try:
    AnswerWithCitaton.model_validate(
        {
            "question": "What is the capital of France?",
            "answer": [
                {"body": "Paris", "substring_quote": "Paris is the capital of France"},
            ],
        },
        context={
            "text_chunks": {
                1: "Jason is a pirate",
                2: "Paris is not the capital of France",
                3: "Irrelevant data",
            }
        },
    )
except ValidationError as e:
    print(e)
"""
answer.0.substring_quote
  Value error, Could not find substring_quote `Paris is the capital of France` in contexts [type=value_error, input_value='Paris is the capital of France', input_type=str]
    For further information visit https://errors.pydantic.dev/2.4/v/value_error
"""


""" 
Example 2) Using an LLM to verify if a 
"""


class Validation(BaseModel):
    """
    Verification response from the LLM,
    the error message should be detailed if the is_valid is False
    but keep it to less than 100 characters, reference specific
    attributes that you are comparing, use `...` is the string is too long
    """

    is_valid: bool
    error_messages: Optional[str] = Field(None, description="Error messages if any")


class Statements(BaseModel):
    body: str
    substring_quote: str

    @model_validator(mode="after")
    def substring_quote_exists(self, info: ValidationInfo):
        context = info.context.get("text_chunks", None)

        resp: Validation = client.chat.completions.create(
            response_model=Validation,
            messages=[
                {
                    "role": "user",
                    "content": f"Does the following citation exist in the following context?\n\nCitation: {self.substring_quote}\n\nContext: {context}",
                }
            ],
            model="gpt-3.5-turbo",
        )

        if resp.is_valid:
            return self

        raise ValueError(resp.error_messages)


class AnswerWithCitaton(BaseModel):
    question: str
    answer: list[Statements]


resp = AnswerWithCitaton.model_validate(
    {
        "question": "What is the capital of France?",
        "answer": [
            {"body": "Paris", "substring_quote": "Paris is the capital of France"},
        ],
    },
    context={
        "text_chunks": {
            1: "Jason is a pirate",
            2: "Paris is the capital of France",
            3: "Irrelevant data",
        }
    },
)
# output: notice that there are no errors
print(resp.model_dump_json(indent=2))
{
    "question": "What is the capital of France?",
    "answer": [{"body": "Paris", "substring_quote": "Paris is the capital of France"}],
}

# Now we change the text chunk to something else, and we get an error
try:
    AnswerWithCitaton.model_validate(
        {
            "question": "What is the capital of France?",
            "answer": [
                {"body": "Paris", "substring_quote": "Paris is the capital of France"},
            ],
        },
        context={
            "text_chunks": {
                1: "Jason is a pirate",
                2: "Paris is not the capital of France",
                3: "Irrelevant data",
            }
        },
    )
except ValidationError as e:
    print(e)
""" 
1 validation error for AnswerWithCitaton
answer.0
  Value error, Citation not found in context [type=value_error, input_value={'body': 'Paris', 'substr... the capital of France'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.4/v/value_error
"""

# Example 3) Using an LLM to verify if the citations and the answers are all aligned


# we keep the same model as above for Statements, but we add a new model for the answer
# that also verifies that the citations are aligned with the answers
class AnswerWithCitaton(BaseModel):
    question: str
    answer: list[Statements]

    @model_validator(mode="after")
    def validate_answer(self, info: ValidationInfo):
        context = info.context.get("text_chunks", None)

        resp: Validation = client.chat.completions.create(
            response_model=Validation,
            messages=[
                {
                    "role": "user",
                    "content": f"Does the following answers match the question and the context?\n\nQuestion: {self.question}\n\nAnswer: {self.answer}\n\nContext: {context}",
                }
            ],
            model="gpt-3.5-turbo",
        )

        if resp.is_valid:
            return self

        raise ValueError(resp.error_messages)


""" 
Using LLMs for citation verification is inefficient during runtime. 
However, we can utilize them to create a dataset consisting only of accurate responses 
where citations must be valid (as determined by LLM, fuzzy text search, etc.). 

This approach would require an initial investment during data generation to obtain 
a finely-tuned model for improved citation.
"""
try:
    AnswerWithCitaton.model_validate(
        {
            "question": "What is the capital of France?",
            "answer": [
                {"body": "Texas", "substring_quote": "Paris is the capital of France"},
            ],
        },
        context={
            "text_chunks": {
                1: "Jason is a pirate",
                2: "Paris is the capital of France",
                3: "Irrelevant data",
            }
        },
    )
except ValidationError as e:
    print(e)
""" 
1 validation error for AnswerWithCitaton
  Value error, The answer does not match the question and context [type=value_error, input_value={'question': 'What is the...he capital of France'}]}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.4/v/value_error
"""
