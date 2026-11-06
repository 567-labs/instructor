import pytest
from itertools import product
from pydantic import BaseModel
import instructor
from ..util import models, modes


class UserDetails(BaseModel):
    name: str
    age: int


# Lists for models, test data, and modes
test_data = [
    ("Jason is 10", "Jason", 10),
    ("Alice is 25", "Alice", 25),
    ("Bob is 35", "Bob", 35),
]


@pytest.mark.parametrize("model, data, mode", product(models, test_data, modes))
def test_extract(model, data, mode):
    sample_data, expected_name, expected_age = data

    client = instructor.from_provider(model=f"google/{model}", mode=mode)

    # Calling the extract function with the provided model, sample data, and mode
    response = client.chat.completions.create(
        response_model=UserDetails,
        messages=[
            {"role": "user", "content": sample_data},
        ],
    )

    # Assertions
    assert response.name == expected_name, (
        f"Expected name {expected_name}, got {response.name}"
    )
    assert response.age == expected_age, (
        f"Expected age {expected_age}, got {response.age}"
    )
