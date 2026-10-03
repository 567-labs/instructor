"""OpenAI-specific schema helpers."""

from __future__ import annotations

import functools
from copy import deepcopy
from typing import Any

from docstring_parser import parse
from pydantic import BaseModel


@functools.lru_cache(maxsize=256)
def generate_openai_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate an OpenAI function schema from a Pydantic model."""
    schema = model.model_json_schema()
    # Recursive models put the root in $defs. Pydantic 2.8 wraps its reference
    # in a single allOf instead of emitting $ref directly. Resolve only that
    # root, retaining $defs and nested references to avoid expanding cycles.
    ref = schema.get("$ref")
    ref_key = "$ref"
    all_of = schema.get("allOf", [])
    if ref is None and len(all_of) == 1 and set(all_of[0]) == {"$ref"}:
        ref = all_of[0]["$ref"]
        ref_key = "allOf"
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        definition = ref[len("#/$defs/") :].replace("~1", "/").replace("~0", "~")
        schema = {
            **deepcopy(schema["$defs"][definition]),
            **{key: value for key, value in schema.items() if key != ref_key},
        }

    docstring = parse(model.__doc__ or "")
    parameters = {k: v for k, v in schema.items() if k not in ("title", "description")}

    for param in docstring.params:
        if (name := param.arg_name) in parameters["properties"] and (
            description := param.description
        ):
            if "description" not in parameters["properties"][name]:
                parameters["properties"][name]["description"] = description

    # Reuse Pydantic's own required set, which excludes any field that has a
    # default -- whether that default is a plain value (``default=``) or a
    # ``default_factory=``. Deriving it from the presence of a ``"default"``
    # key in each property missed default_factory fields (whose defaults are
    # never emitted into the JSON schema) and wrongly marked them required.
    parameters["required"] = sorted(schema.get("required", []))

    if "description" not in schema:
        schema["description"] = (
            docstring.short_description
            or f"Correctly extracted `{model.__name__}` with all "
            "the required parameters with correct types"
        )

    return {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": parameters,
    }
