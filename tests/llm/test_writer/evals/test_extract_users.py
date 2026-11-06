import pytest
from itertools import product
from pydantic import BaseModel
from writerai import Writer
import instructor
from ..util import models, modes


class UserDetails(BaseModel):
    first_name: str
    age: int


test_data = [
    ("Jason is 10", "Jason", 10),
    ("Alice is 25", "Alice", 25),
    ("Bob is 35", "Bob", 35),
]


@pytest.mark.parametrize("model, data, mode", product(models, test_data, modes))
def test_writer_extract(
    model: str, data: list[tuple[str, str, int]], mode: instructor.Mode
):
    client = instructor.from_writer(client=Writer(), mode=mode)

    sample_data, expected_name, expected_age = data

    response = client.chat.completions.create(
        model=model,
        response_model=UserDetails,
        messages=[
            {"role": "user", "content": sample_data},
        ],
    )

    assert response.first_name == expected_name, (
        f"Expected name {expected_name}, got {response.first_name}"
    )
    assert response.age == expected_age, (
        f"Expected age {expected_age}, got {response.age}"
    )
