---
authors:
- jxnl
categories:
- Validation
- Pydantic
- LLMs
comments: true
date: 2025-05-20
description: Learn how semantic validation with LLMs can ensure your structured outputs meet complex, subjective, and contextual criteria beyond what traditional rule-based validation can achieve.
draft: false
tags:
- Semantic Validation
- Structured Outputs
- LLM Validator
- Pydantic
- Data Quality
---

# Understanding Semantic Validation with Structured Outputs

> Semantic validation uses LLMs to evaluate content against complex, subjective, and contextual criteria that would be difficult to implement with traditional rule-based validation approaches.

As LLMs become increasingly integrated into production systems, ensuring the quality and safety of their outputs is paramount. Traditional validation methods relying on explicit rules can't keep up with the complexity and nuance of natural language. With the release of Instructor's semantic validation capabilities, we now have a powerful way to validate structured outputs against sophisticated criteria.

<!-- more -->

## Beyond Rule-Based Validation

Traditional validation approaches focus on verifying that data conforms to certain rules-ensuring that:

- A field has the correct type (`int`, `str`, etc.)
- A value falls within predefined ranges (e.g., `age >= 0`)
- A pattern matches expected formats (e.g., email regex)

These approaches work well for structured data with clear constraints but fall short when validating natural language against less precise criteria like:

- "Content must be family-friendly"
- "Description must be professional and free of hyperbole"
- "Criticism must be constructive and respectful"
- "Message must adhere to community guidelines"

This is where semantic validation with LLMs comes in.

## What is Semantic Validation?

Semantic validation uses an LLM to interpret and evaluate text against natural language criteria. Instead of writing explicit rules, you express validation requirements in plain language, and the LLM determines whether content meets those requirements.

Let's see how this works with Instructor's `llm_validator`:

```python
from typing import Annotated
from pydantic import BaseModel, BeforeValidator
import instructor
from instructor import llm_validator

# Initialize client
client = instructor.from_provider("openai/gpt-5-nano")


class ProductDescription(BaseModel):
    name: str
    description: Annotated[
        str,
        BeforeValidator(
            llm_validator(
                """The description must be:
                1. Professional and factual
                2. Free of excessive hyperbole or unsubstantiated claims
                3. Between 50-200 words in length
                4. Written in third person (no "you" or "your")
                5. Free of spelling and grammar errors""",
                client=client,
            )
        ),
    ]
```

What makes this approach powerful is that we're leveraging the LLM's understanding of language and context to perform validation that would be extremely difficult to implement with traditional approaches.

## When to Use Semantic Validation

Semantic validation shines in situations where:

1. **Criteria is complex or subjective**: "Ensure this content is respectful" requires understanding nuance that's difficult to capture in rules.

2. **Context matters**: "The summary must accurately reflect the key findings" requires comparing multiple pieces of content.

3. **The rules are constantly evolving**: Harmful content strategies change as bad actors adapt, making static rules obsolete quickly.

4. **Human-like judgment is required**: "This product description should be compelling without being misleading" requires nuanced evaluation.

## Real-World Examples

### Content Moderation

One of the most obvious applications is content moderation. Companies need to ensure user-generated content meets community guidelines without being overly restrictive:

```python
class UserComment(BaseModel):
    user_id: str
    content: Annotated[
        str,
        BeforeValidator(
            llm_validator(
                """Content must comply with community guidelines:
                - No hate speech, harassment, or discrimination
                - No explicit sexual or violent content
                - No promotion of illegal activities
                - No sharing of personal information
                - No spamming or excessive self-promotion""",
                client=client,
            )
        ),
    ]
```

### Tone and Style Enforcement

Organizations often need to maintain a consistent tone and style in their communications:

```python
class CompanyAnnouncement(BaseModel):
    title: str
    content: Annotated[
        str,
        BeforeValidator(
            llm_validator(
                "The announcement must maintain a professional, positive tone without being overly informal or using slang",
                client=client,
            )
        ),
    ]
```

### Fact-Checking

For applications where factual accuracy is critical:

```python
class FactCheckedClaim(BaseModel):
    claim: str
    is_accurate: bool
    supporting_evidence: list[str]

    @classmethod
    def validate_claim(cls, text: str) -> "FactCheckedClaim":
        return client.create(
            response_model=cls,
            messages=[
                {
                    "role": "system",
                    "content": "You are a fact-checking system. Assess the factual accuracy of the claim.",
                },
                {"role": "user", "content": "Fact check this claim: {{ claim }}"},
            ],
            context={"claim": text},
        )
```

## Beyond Field Validation: Model-Level Semantic Validation

While field-level validation is powerful, sometimes we need to validate relationships between fields. This is where model-level semantic validation becomes useful:

```python
class Report(BaseModel):
    title: str
    summary: str
    key_findings: list[str]

    @model_validator(mode='after')
    def validate_consistency(self):
        # Semantic validation at the model level using Jinja templating
        validation_result = client.create(
            response_model=Validator,
            messages=[
                {
                    "role": "system",
                    "content": "Validate that the summary accurately reflects the key findings.",
                },
                {
                    "role": "user",
                    "content": """
                        Please validate if this summary accurately reflects the key findings:

                        Title: {{ title }}
                        Summary: {{ summary }}

                        Key findings:
                        {% for finding in findings %}
                        - {{ finding }}
                        {% endfor %}

                        Evaluate for consistency, completeness, and accuracy.
                    """,
                },
            ],
            context={
                "title": self.title,
                "summary": self.summary,
                "findings": self.key_findings,
            },
        )

        if not validation_result.is_valid:
            raise ValueError(f"Consistency error: {validation_result.reason}")

        return self
```

## Technical Implementation

Under the hood, the `llm_validator` uses a special `Validator` model that determines whether content meets the criteria and provides detailed error messages when it doesn't:

```python
class Validator(BaseModel):
    is_valid: bool
    reason: Optional[str] = None
    fixed_value: Optional[str] = None
```

When validation fails, the reason field contains a detailed explanation, which is perfect for both developers debugging issues and for automatic retry mechanisms.

## Self-Healing with Retries

One of the most powerful features of Instructor's validation system is its ability to automatically retry with error context:

```python
try:
    product = client.create(
        response_model=ProductDescription,
        messages=[
            {"role": "system", "content": "Generate a product description."},
            {
                "role": "user",
                "content": "Create a description for UltraClean 9000 Washing Machine",
            },
        ],
        max_retries=2,  # Automatically retry up to 2 times with error context
    )
    print("Success:", product.model_dump_json(indent=2))
except Exception as e:
    print(f"Failed after retries: {e}")
    #> Failed after retries: name 'client' is not defined
```

With `max_retries` set, if the initial response fails validation, Instructor will automatically send the error context back to the LLM, giving it a chance to correct the issue. This creates a self-healing system that can recover from validation failures without developer intervention.

## Performance and Cost Considerations

Semantic validation adds an additional API call for each validation, which impacts:

1. **Latency**: Each validation requires an LLM inference
2. **Cost**: More API calls mean higher usage costs
3. **Reliability**: Depends on LLM API availability

For high-throughput applications, consider these strategies:

- **Batch validations**: Validate multiple items in a single call where possible
- **Strategic placement**: Apply semantic validation at critical points rather than everywhere
- **Caching**: Cache validation results for identical or similar content
- **Use the right model**: `gpt-4o-mini` or similar models offer a good balance of capability and cost for many validation scenarios

## Building a Layered Validation Strategy

The most robust approach combines traditional validation with semantic validation:

1. **Type validation**: Use Pydantic's built-in type validation as your first defense
2. **Rule-based validation**: Apply explicit rules where they make sense
3. **Semantic validation**: Reserve LLM-based validation for complex criteria

This layered approach ensures you get the benefits of semantic validation without unnecessary API calls for simple validations.

## Advanced Applications

### Custom Guardrails Framework

You can build a comprehensive guardrails framework by combining semantic validators:

```python
def create_guarded_model(base_class, guardrails):
    """Create a model with multiple semantic guardrails applied."""
    validators = {}

    for field_name, criteria in guardrails.items():
        validators[field_name] = Annotated[
            str, BeforeValidator(llm_validator(criteria, client=client))
        ]

    return create_model(
        f"Guarded{base_class.__name__}", __base__=base_class, **validators
    )


# Usage
guardrails = {
    "title": "Must be concise, descriptive, and free of clickbait",
    "content": "Must follow community guidelines and be respectful",
}

GuardedPost = create_guarded_model(Post, guardrails)
```

### Contextual Validation with External References

For validations that require external knowledge:

```python
class LegalCompliance(BaseModel):
    document: str
    compliance_status: Annotated[
        str,
        BeforeValidator(
            llm_validator(
                """Check if this document complies with the provided guidelines.
                Guidelines: {{ guidelines }}""",
                client=client,
            )
        ),
    ]


# Usage
result = client.create(
    response_model=LegalCompliance,
    messages=[{"role": "user", "content": "Check this document: " + document_text}],
    context={"guidelines": company_legal_guidelines},
)
```

## Conclusion

Semantic validation represents a significant advancement in ensuring the quality and safety of LLM outputs. By combining the flexibility of natural language criteria with the structured validation of Pydantic, we can build systems that are both powerful and safe.

As these techniques mature, we can expect to see semantic validation become a standard part of AI application development, especially in regulated industries where output quality is critical.

To get started with semantic validation in your projects, check out the [Semantic Validation documentation](../../concepts/semantic_validation.md) and explore the various examples and patterns.

This approach isn't just a technical improvement-it's a fundamental shift in how we think about validation, moving from rigid rules to intelligent understanding of content and context.

## Related Documentation
- [Validation Fundamentals](../../concepts/validation.md) - Core validation concepts
- [Semantic Validation](../../concepts/semantic_validation.md) - Using LLMs for validation

## See Also
- [Validation Deep Dive](validation-part1.md) - Foundation validation concepts
- [Anthropic Prompt Caching](anthropic-prompt-caching.md) - Optimize validation costs
- [Monitoring with Logfire](logfire.md) - Track validation performance
