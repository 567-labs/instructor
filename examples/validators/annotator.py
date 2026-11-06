from typing import Annotated
from pydantic import BaseModel, ValidationError
from pydantic.functional_validators import AfterValidator


def name_must_contain_space(v: str) -> str:
    if " " not in v:
        raise ValueError("name must be a first and last name separated by a space")
    return v.lower()


class UserDetail(BaseModel):
    age: int
    name: Annotated[str, AfterValidator(name_must_contain_space)]


# Example 1) Valid input, notice that the name is lowercased
person: UserDetail = UserDetail(age=29, name="Jason Liu")
print(person.model_dump_json(indent=2))
"""
{
    "age": 29,
    "name": "jason liu"
}
"""

# Example 2) Invalid input, we'll get a validation error
# In the future this validation error will be raised by the API and
# used by the LLM to generate a better response
try:
    person: UserDetail = UserDetail(age=29, name="Jason")
except ValidationError as e:
    print(e)
    """
    1 validation error for UserDetail
    name
        Value error, name must be a first and last name separated by a space [type=value_error, input_value='Jason', input_type=str]
        For further information visit https://errors.pydantic.dev/2.3/v/value_error
    """
