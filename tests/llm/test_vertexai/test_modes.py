"""VertexAI-specific tests for mixed content types.

Tests VertexAI's ability to handle mixed content with gm.Part objects.
"""

from itertools import product
from pydantic import BaseModel
import vertexai.generative_models as gm  # type: ignore
import pytest
import instructor

from .util import models, modes


class Item(BaseModel):
    name: str
    price: float


class Order(BaseModel):
    items: list[Item]
    customer: str


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_mixed_content_types(model, mode):
    client = instructor.from_vertexai(gm.GenerativeModel(model), mode)
    content = [
        "Order Details:",
        gm.Part.from_text("Customer: Alice"),
        gm.Part.from_text("Items:"),
        "Name: Laptop, Price: 999.99",
        "Name: Mouse, Price: 29.99",
    ]

    resp = client.create(
        response_model=Order,
        messages=[
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    assert len(resp.items) == 2
    assert {x.name.lower() for x in resp.items} == {"laptop", "mouse"}
    assert {x.price for x in resp.items} == {999.99, 29.99}
    assert resp.customer.lower() == "alice"
