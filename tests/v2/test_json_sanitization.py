import json

import pytest
from pydantic import BaseModel, ValidationError

from instructor.v2.core.function_calls import _validate_model_from_json


class Answer(BaseModel):
    value: str


@pytest.mark.parametrize("strict", [True, False])
def test_fenced_json_uses_the_same_validation_as_plain_json(strict):
    payload = '{"value": "hello ``` world"}'
    assert _validate_model_from_json(
        Answer, "  ```json\n" + payload + "\n```  ", strict=strict
    ) == Answer(value="hello ``` world")
    assert _validate_model_from_json(list[int], "```\n[1, 2]\n```", strict=strict) == [
        1,
        2,
    ]
    assert (
        _validate_model_from_json(str, '```json\n"hello"\n```', strict=strict)
        == "hello"
    )
    with pytest.raises((ValidationError, json.JSONDecodeError)):
        _validate_model_from_json(Answer, "```json\n{invalid}\n```", strict=strict)
    with pytest.raises((ValidationError, json.JSONDecodeError)):
        _validate_model_from_json(Answer, "", strict=strict)
