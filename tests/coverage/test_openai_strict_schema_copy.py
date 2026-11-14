from copy import deepcopy
from typing import Any

from instructor.v2.providers.openai.schema import make_strict_schema


def test_strict_schema_copy_keeps_shared_and_custom_metadata_isolated() -> None:
    class Metadata(dict[str, Any]):
        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            return Metadata(copied=deepcopy(self["value"], memo))

    shared = {"values": [True, None, 1, 2.5, "example"]}
    metadata = {
        "first": shared,
        "second": (Metadata(value=shared),),
        b"custom key": shared,
    }
    metadata["cycle"] = metadata
    source = {"type": "object", "properties": {}, "examples": [metadata]}

    result = make_strict_schema(source)
    copied = result["examples"][0]

    assert result["additionalProperties"] is False
    assert "additionalProperties" not in source
    assert copied is not metadata
    assert copied["cycle"] is copied
    assert isinstance(copied["second"][0], Metadata)
    assert copied["first"] is copied["second"][0]["copied"] is copied[b"custom key"]
    assert copied["first"] is not shared
    copied["first"]["values"].append("new")
    assert shared["values"] == [True, None, 1, 2.5, "example"]
