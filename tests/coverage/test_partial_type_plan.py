from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from instructor.v2.dsl.partial import Partial


def test_partial_model_list_preserves_null_until_final_validation() -> None:
    class Item(BaseModel):
        count: int

    class Envelope(BaseModel):
        items: list[Item]

    partial_model = cast(Any, Partial[Envelope])
    stream = partial_model.model_from_chunks(
        iter(['{"items":[null,{"count":"2"}', "]}"])
    )

    snapshot = next(stream)
    assert snapshot.items[0] is None
    assert isinstance(snapshot.items[1], Item)
    assert snapshot.items[1].count == "2"
    with pytest.raises(ValidationError, match=r"items\.0"):
        next(stream)
