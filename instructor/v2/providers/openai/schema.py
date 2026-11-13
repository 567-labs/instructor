"""OpenAI-specific schema helpers."""

from __future__ import annotations

import functools
from copy import deepcopy
from typing import Any, cast

from docstring_parser import parse
from pydantic import BaseModel


@functools.lru_cache(maxsize=256)
def generate_openai_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate an OpenAI function schema from a Pydantic model."""
    schema = model.model_json_schema()
    docstring = parse(model.__doc__ or "")
    parameters = {k: v for k, v in schema.items() if k not in ("title", "description")}

    properties = parameters.get("properties")
    if isinstance(properties, dict):
        for param in docstring.params:
            if (name := param.arg_name) in properties and (
                description := param.description
            ):
                field_spec = properties[name]
                if isinstance(field_spec, dict) and "description" not in field_spec:
                    field_spec["description"] = description

    # Reuse Pydantic's own required set, which excludes any field that has a
    # default -- whether that default is a plain value (``default=``) or a
    # ``default_factory=``. Deriving it from the presence of a ``"default"``
    # key in each property missed default_factory fields (whose defaults are
    # never emitted into the JSON schema) and wrongly marked them required.
    if parameters.get("type") == "object" or isinstance(properties, dict):
        parameters["required"] = sorted(schema.get("required", []))
    elif "required" in schema:
        parameters["required"] = sorted(schema.get("required", []))

    model_name = getattr(model, "__name__", None) or type(model).__name__

    if "description" not in schema:
        schema["description"] = (
            docstring.short_description
            or f"Correctly extracted `{model_name}` with all "
            "the required parameters with correct types"
        )

    title = schema.get("title") or model_name

    return {
        "name": title,
        "description": schema["description"],
        "parameters": parameters,
    }


def make_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copied OpenAI strict schema, preserving recursive references."""
    from openai.lib._pydantic import resolve_ref

    strict_schema = deepcopy(schema)

    def normalize(
        schema: dict[str, Any] | bool,
        active_refs: frozenset[str] = frozenset(),
    ) -> None:
        # The SDK strict helper rejects boolean schemas, which batch callers
        # already use. Traverse schema keywords only, leaving metadata intact.
        if isinstance(schema, bool):
            return
        properties = schema.get("properties")
        additional = schema.get("additionalProperties")
        if (
            isinstance(additional, dict)
            or (additional is True and not properties)
            or (
                schema.get("type") == "object"
                and "properties" not in schema
                and additional is None
            )
            or schema.get("patternProperties")
        ):
            raise ValueError(
                "Arbitrary mapping schemas are not supported by OpenAI "
                "strict mode; use a model with named fields or pass strict=False"
            )
        schema_type = schema.get("type")
        if (
            schema_type == "object"
            or (isinstance(schema_type, list) and "object" in schema_type)
            or isinstance(properties, dict)
        ):
            schema["additionalProperties"] = False
        if isinstance(properties, dict):
            schema["required"] = list(properties)
            for child in properties.values():
                normalize(child, active_refs)
        for keyword in ("$defs", "definitions"):
            for child in schema.get(keyword, {}).values():
                normalize(child, active_refs)
        if "items" in schema:
            normalize(schema["items"], active_refs)
        for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
            for child in schema.get(keyword, []):
                normalize(child, active_refs)
        all_of = schema.get("allOf")
        if (
            isinstance(all_of, list)
            and len(all_of) == 1
            and isinstance(all_of[0], dict)
        ):
            schema.update(all_of[0])
            del schema["allOf"]
        # Match the SDK's nullable-default handling; retain non-null defaults.
        if "default" in schema and schema["default"] is None:
            del schema["default"]
        ref = schema.get("$ref")
        if isinstance(ref, str) and len(schema) > 1 and ref not in active_refs:
            resolved = cast(dict[str, Any], resolve_ref(root=strict_schema, ref=ref))
            # Copy the referenced node, and retain recursive back-references
            # with their metadata instead of expanding indefinitely.
            schema.update({**deepcopy(resolved), **schema})
            del schema["$ref"]
            normalize(schema, active_refs | {ref})

    normalize(strict_schema)
    return strict_schema
