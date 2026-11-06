from pydantic import BaseModel, ValidationError, field_validator


class UserDetail(BaseModel):
    age: int
    name: str

    @field_validator("name", mode="before")
    def name_must_contain_space(cls, v):
        """
        This validator will be called after the default validator,
        and will raise a validation error if the name does not contain a space.
        then it will set the name to be lower case
        """
        if " " not in v:
            raise ValueError("name be a first and last name separated by a space")
        return v.lower()


# Example 1) Valid input, notice that the name is lowercased
person = UserDetail(age=29, name="Jason Liu")
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
    person = UserDetail(age=29, name="Jason")
except ValidationError as e:
    print(e)
    """
    1 validation error for UserDetail 
        name
    Value error, must contain a space [type=value_error, input_value='Jason', input_type=str]
        For further information visit https://errors.pydantic.dev/2.3/v/value_error
    """
