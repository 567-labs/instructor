from pydantic import BaseModel, ValidationError, field_validator, ValidationInfo


class AnswerWithCitation(BaseModel):
    answer: str
    citation: str

    @field_validator("citation")
    @classmethod
    def remove_stopwords(cls, v: str, info: ValidationInfo):
        context = info.context
        if context:
            text_chunks = context.get("text_chunk")
            if v not in text_chunks:
                raise ValueError(f"Citation `{v}` not found in text chunks")
        return v


try:
    AnswerWithCitation.model_validate(
        {"answer": "Jason is a cool guy", "citation": "Jason is cool"},
        context={"text_chunk": "Jason is just a guy"},
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for AnswerWithCitation
    citation
    Value error, Citation `Jason is cool`` not found in text chunks [type=value_error, input_value='Jason is cool', input_type=str]
        For further information visit https://errors.pydantic.dev/2.4/v/value_error
    """
