from itertools import product
from writerai import Writer

from pydantic import BaseModel, Field
import pytest

import instructor
from instructor import Instructor

from ..util import models, modes


class Property(BaseModel):
    key: str
    value: str
    resolved_absolute_value: str


class Entity(BaseModel):
    id: int = Field(
        ...,
        description="Unique identifier for the entity, used for deduplication, design a scheme allows multiple entities",
    )
    subquote_string: list[str] = Field(
        ...,
        description="Correctly resolved value of the entity, if the entity is a reference to another entity, this should be the id of the referenced entity, include a few more words before and after the value to allow for some context to be used in the resolution",
    )
    entity_title: str
    properties: list[Property] = Field(
        ..., description="List of properties of the entity"
    )
    dependencies: list[int] = Field(
        ...,
        description="List of entity ids that this entity depends  or relies on to resolve it",
    )


class DocumentExtraction(BaseModel):
    entities: list[Entity] = Field(
        ...,
        description="Body of the answer, each fact should be its separate object with a body and a list of sources",
    )


def ask_ai(content: str, model: str, client: Instructor) -> DocumentExtraction:
    resp: DocumentExtraction = client.chat.completions.create(
        model=model,
        response_model=DocumentExtraction,
        messages=[
            {
                "role": "system",
                "content": "You are a perfect entity resolution system that extracts facts from the document. Extract and resolve a list of entities from the following document:",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        max_retries=4,
    )  # type: ignore
    return resp


content = """
Sample Legal Contract
Agreement Contract

This Agreement is made and entered into on 2020-01-01 by and between Company A ("the Client") and Company B ("the Service Provider").

Article 1: Scope of Work

The Service Provider will deliver the software product to the Client 30 days after the agreement date.

Article 2: Payment Terms

The total payment for the service is $50,000.
An initial payment of $10,000 will be made within 7 days of the the signed date.
The final payment will be due 45 days after [SignDate].

Article 3: Confidentiality

The parties agree not to disclose any confidential information received from the other party for 3 months after the final payment date.

Article 4: Termination

The contract can be terminated with a 30-day notice, unless there are outstanding obligations that must be fulfilled after the [DeliveryDate].
"""


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_extract(model: str, mode: instructor.Mode):
    client = instructor.from_writer(client=Writer(), mode=mode)

    extract = ask_ai(content=content, model=model, client=client)
    assert len(extract.entities) > 0
