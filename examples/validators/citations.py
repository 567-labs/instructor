from typing import Annotated
from pydantic import BaseModel, ValidationError, ValidationInfo, AfterValidator
from openai import OpenAI
import instructor

client = instructor.from_openai(OpenAI())


def citation_exists(v: str, info: ValidationInfo):
    context = info.context
    if context:
        context = context.get("text_chunk")
        if v not in context:
            raise ValueError(f"Citation `{v}` not found in text")
    return v


Citation = Annotated[str, AfterValidator(citation_exists)]


class AnswerWithCitation(BaseModel):
    answer: str
    citation: Citation


try:
    q = "Are blue berries high in protein?"
    text_chunk = """
    Blueberries are a good source of vitamin K.
    They also contain vitamin C, fibre, manganese and other antioxidants (notably anthocyanins).    
    """

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=AnswerWithCitation,
        messages=[
            {
                "role": "user",
                "content": f"Answer the question `{q}` using the text chunk\n`{text_chunk}`",
            },
        ],
        validation_context={"text_chunk": text_chunk},
    )  # type: ignore
    print(resp.model_dump_json(indent=2))
except ValidationError as e:
    print(e)
