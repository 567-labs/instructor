"""OpenAI-specific schema helpers."""

from __future__ import annotations

import copy
import functools
from typing import Any

from docstring_parser import parse
from pydantic import BaseModel

__all__ = ["generate_openai_schema", "make_strict_schema"]


@functools.lru_cache(maxsize=256)
def generate_openai_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate an OpenAI function schema from a Pydantic model."""
    schema = model.model_json_schema()
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


def make_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively enforce OpenAI Structured Outputs strict JSON Schema constraints.

    - Sets 'additionalProperties: False' on all object schemas (root, definitions, nested).
    - Populates 'required' with all property keys for schemas with 'properties'.
    - Strips 'default: None' from property schemas to comply with OpenAI strict schema rules.
    - Resolves and inlines '$ref' when sibling properties (like description) are present.
    - Recursively processes nested schemas in 'properties', '$defs', 'definitions',
      'items', 'prefixItems', 'anyOf', 'allOf', 'oneOf', 'not', and 'dependentSchemas'.
    """
    root = copy.deepcopy(schema)

    def _resolve_ref(ref: str) -> dict[str, Any] | None:
        if not ref.startswith("#/"):
            return None
        parts = ref[2:].split("/")
        curr: Any = root
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                return None
        return curr if isinstance(curr, dict) else None

    def _normalize(node: Any) -> Any:
        if not isinstance(node, dict):
            return node

        if "$ref" in node and len(node) > 1:
            ref = node["$ref"]
            if isinstance(ref, str):
                resolved = _resolve_ref(ref)
                if isinstance(resolved, dict):
                    resolved_copy = _normalize(copy.deepcopy(resolved))
                    del node["$ref"]
                    for k, v in resolved_copy.items():
                        if k not in node:
                            node[k] = v

        if "default" in node and node["default"] is None:
            del node["default"]

        if node.get("type") == "object" or "properties" in node:
            if node.get("additionalProperties") is not False:
                node["additionalProperties"] = False

        if "properties" in node and isinstance(node["properties"], dict):
            node["required"] = list(node["properties"].keys())
            for prop_name, prop_schema in node["properties"].items():
                node["properties"][prop_name] = _normalize(prop_schema)

        for def_key in ("$defs", "definitions", "dependentSchemas"):
            if def_key in node and isinstance(node[def_key], dict):
                for def_name, def_schema in node[def_key].items():
                    node[def_key][def_name] = _normalize(def_schema)

        if "items" in node:
            if isinstance(node["items"], dict):
                node["items"] = _normalize(node["items"])
            elif isinstance(node["items"], list):
                node["items"] = [_normalize(item) for item in node["items"]]

        if "prefixItems" in node and isinstance(node["prefixItems"], list):
            node["prefixItems"] = [_normalize(item) for item in node["prefixItems"]]

        if "allOf" in node and isinstance(node["allOf"], list):
            if len(node["allOf"]) == 1:
                single = _normalize(node.pop("allOf")[0])
                if isinstance(single, dict):
                    for k, v in single.items():
                        if k not in node:
                            node[k] = v
            else:
                node["allOf"] = [_normalize(sub) for sub in node["allOf"]]

        for comb_key in ("anyOf", "oneOf"):
            if comb_key in node and isinstance(node[comb_key], list):
                node[comb_key] = [_normalize(sub) for sub in node[comb_key]]

        if "not" in node and isinstance(node["not"], dict):
            node["not"] = _normalize(node["not"])

        if "patternProperties" in node and isinstance(node["patternProperties"], dict):
            for pat_name, pat_schema in node["patternProperties"].items():
                node["patternProperties"][pat_name] = _normalize(pat_schema)

        return node

    return _normalize(root)
