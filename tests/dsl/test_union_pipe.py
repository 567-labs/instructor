import sys
import pytest
from typing import Union

import instructor
from instructor.dsl.partial import _process_generic_arg
from pydantic import BaseModel


class UserWithUnion(BaseModel):
    name: str
    age: int
    category: Union[str, None]


class UserWithPipe(BaseModel):
    name: str
    age: int
    category: str | None


def test_process_generic_arg_union():
    """Test that _process_generic_arg correctly handles Union[str, None]."""
    annotation = UserWithUnion.__annotations__["category"]
    processed = _process_generic_arg(annotation)
    assert processed is not None


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Union pipe syntax requires Python 3.10+")
def test_process_generic_arg_pipe():
    """Test that _process_generic_arg correctly handles str | None."""
    annotation = UserWithPipe.__annotations__["category"]
    processed = _process_generic_arg(annotation)
    assert processed is not None
