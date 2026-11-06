import instructor
from pydantic import BaseModel
from .util import models, modes
import pytest
from itertools import product
import vertexai.generative_models as gm


class User(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model, mode, is_list", product(models, modes, [True, False]))
def test_format_string(model, mode, is_list):
    client = instructor.from_vertexai(
        gm.GenerativeModel(model),
        mode=mode,
    )

    content = (
        [gm.Part.from_text("Extract {{name}} is {{age}} years old.")]
        if is_list
        else "Extract {{name}} is {{age}} years old."
    )

    # note that client.chat.completions.create will also work
    resp = client.messages.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        response_model=User,
        context={"name": "Jason", "age": 25},
    )

    assert isinstance(resp, User)
    assert resp.name == "Jason"
    assert resp.age == 25
