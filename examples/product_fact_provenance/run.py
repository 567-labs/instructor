"""Extract product facts and keep unsupported claims visible for review."""

import json
from collections.abc import Mapping

import instructor

from examples.product_fact_provenance import ProductExtraction


SOURCE_DOCUMENTS = {
    "supplier-sheet:row-7": "Material: 304 stainless steel.",
    "package-photo:ocr": "Capacity: 750 ml.",
    "supplier-copy:paragraph-2": "Designed for everyday travel.",
}


def extract_product_facts(sources: Mapping[str, str]) -> ProductExtraction:
    """Extract facts and validate their evidence against the supplied documents."""

    client = instructor.from_provider("openai/gpt-5-nano")
    return client.create(
        response_model=ProductExtraction,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract product facts only from the supplied sources. For every "
                    "value, copy an exact supporting quote and its source ID. Use null "
                    "and an empty evidence list when a value is not supported."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(sources, ensure_ascii=False, indent=2),
            },
        ],
        context={"sources": sources},
    )


if __name__ == "__main__":
    result = extract_product_facts(SOURCE_DOCUMENTS)
    print(result.model_dump_json(indent=2))
