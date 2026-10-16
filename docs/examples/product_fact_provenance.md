---
title: Product Fact Extraction with Provenance
description: Keep source IDs and exact quotes for product facts, then mark unsupported AI output for human review.
---

# Product Fact Extraction with Provenance

Structured output proves that an LLM returned the right data shape. It does not
prove that a material, capacity, certification, or product claim is true.

This example adds a conservative check after generation. Each product fact keeps
a stable source ID and an exact quote. Local validation then checks that:

1. The source ID exists.
2. The quote appears exactly in that source.
3. The extracted value appears in the quote after case and whitespace normalization.

Unsupported facts stay in the result with an `unverified` status. This makes them
easy to send to a human reviewer instead of silently dropping them.

## Validate facts without another LLM call

The example models compute `verified`, `status`, and `confidence` locally. These
fields are not requested from the model, so the model cannot mark its own claim
as verified. Here, `confidence` means the fraction of evidence items that passed
the deterministic checks. It is not a probability supplied by the LLM.

```python
from examples.product_fact_provenance import ProductExtraction

sources = {
    "supplier-sheet:row-7": "Material: 304 stainless steel.",
    "package-photo:ocr": "Capacity: 750 ml.",
}

extraction = ProductExtraction.model_validate(
    {
        "facts": [
            {
                "field_name": "material",
                "value": "304 stainless steel",
                "evidence": [
                    {
                        "source_id": "supplier-sheet:row-7",
                        "quote": "Material: 304 stainless steel.",
                    }
                ],
            },
            {
                "field_name": "waterproof_rating",
                "value": "IPX7",
                "evidence": [],
            },
        ]
    },
    context={"sources": sources},
)

assert extraction.facts[0].status == "verified"
assert extraction.facts[0].confidence == 1.0
assert extraction.facts[1].status == "unverified"
assert extraction.facts[1].confidence == 0.0
```

## Use the models with Instructor

Pass the same source mapping to the prompt and to Instructor's validation
`context`. Ask the model to copy exact quotes and use `null` when the sources do
not support a value.

```python
import json
from collections.abc import Mapping

import instructor

from examples.product_fact_provenance import ProductExtraction


def extract_product_facts(sources: Mapping[str, str]) -> ProductExtraction:
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
```

The complete runnable example is in
[`examples/product_fact_provenance`](https://github.com/567-labs/instructor/tree/main/examples/product_fact_provenance).

## Limits

This check is intentionally strict. It catches missing sources, invented quotes,
and values that do not appear in the cited text. It does not prove that a source
document is correct, resolve conflicting supplier documents, or verify a
paraphrase. Those cases should stay unverified until a human or a separate
domain-specific check reviews them.

## See Also

- [Exact Citations for RAG](./exact_citations.md)
- [Validation](../concepts/validation.md)
- [CitationMixin](../concepts/citation.md)
