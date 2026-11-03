import pytest
from pydantic import BaseModel

from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.dsl.simple_type import is_simple_type


class User(BaseModel):
    name: str


def test_is_simple_type_does_not_hash_the_input():
    assert is_simple_type(User()) is False


def test_instance_response_model_is_rejected_with_a_clear_error():
    with pytest.raises(TypeError, match="must be a class, not an instance"):
        prepare_response_model(User())


def test_scalar_and_model_classes_are_unchanged():
    assert is_simple_type(str) is True
    assert is_simple_type(User) is False
    assert prepare_response_model(str) is str
