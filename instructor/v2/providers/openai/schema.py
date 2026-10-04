"""OpenAI-specific schema helpers."""

from __future__ import annotations

import functools
from collections import Counter
from typing import Any

from docstring_parser import parse
from pydantic import AliasChoices, BaseModel


@functools.lru_cache(maxsize=256)
def generate_openai_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate an OpenAI function schema from a Pydantic model."""
    schema = model.model_json_schema()
    docstring = parse(model.__doc__ or "")
    parameters = {k: v for k, v in schema.items() if k not in ("title", "description")}

    property_names: dict[str, str] = {}
    serialization = (
        model.model_config.get("json_schema_mode_override") == "serialization"
    )
    for field_name, field in model.model_fields.items():
        name = field_name
        # Docstrings name Python fields; schemas use their input aliases.
        # Match Pydantic's first single-string choice for AliasChoices.
        alias = field.serialization_alias if serialization else field.validation_alias
        if isinstance(alias, str):
            name = alias
        elif isinstance(alias, AliasChoices):
            for path in alias.convert_to_aliases():
                if len(path) == 1 and isinstance(path[0], str):
                    name = path[0]
                    break
        property_names[field_name] = name
    if serialization:
        for field_name, computed in model.model_computed_fields.items():
            property_names[field_name] = (
                computed.alias if computed.alias is not None else field_name
            )
    property_counts = Counter(property_names.values())

    for param in docstring.params:
        name = property_names.get(param.arg_name, param.arg_name)
        if param.arg_name in property_names and property_counts[name] > 1:
            # Shared aliases may include omitted fields. Do not guess which
            # field owns the emitted property or attach another field's prompt.
            continue

        if name in parameters["properties"] and (description := param.description):
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
