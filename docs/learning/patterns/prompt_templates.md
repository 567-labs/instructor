---
title: Using Prompt Templates with Instructor
description: Learn how to create reusable prompt templates for consistent structured output extraction across different use cases.
---

# Prompt Templates

This guide covers how to use prompt templates with Instructor to create reusable, parameterized prompts for structured data extraction.

## Why Prompt Templates Matter

Good prompts are essential for effective structured data extraction. Prompt templates help you:

1. Create consistent and reusable prompts
2. Parameterize prompts with dynamic values
3. Separate prompt engineering from application logic
4. Standardize prompt patterns for different use cases

## Basic Prompt Templates

The simplest form of a prompt template is a string with placeholders for variables:

```python
from pydantic import BaseModel, Field
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class Person(BaseModel):
    name: str
    age: int
    occupation: str

# Define a template with parameters
prompt_template = """
Extract information about the person mentioned in the following {document_type}:

{content}

Please provide their name, age, and occupation.
"""

# Use the template with specific values
document_type = "email"
content = "Hi team, I'm introducing our new project manager, Sarah Johnson. She's 34 and has been in project management for 8 years."

prompt = prompt_template.format(
    document_type=document_type,
    content=content
)

# Extract structured data using the formatted prompt
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    response_model=Person
)
```

## Using f-strings for Simple Templates

For simple cases, you can use f-strings to create prompt templates:

```python
def extract_person(content, document_type="text"):
    prompt = f"""
    Extract information about the person mentioned in the following {document_type}:

    {content}

    Please provide their name, age, and occupation.
    """

    return client.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_model=Person
    )

# Use the function
person = extract_person(
    "According to his resume, John Smith (42) works as a software developer.",
    document_type="resume"
)
```

## Template Functions

For more complex templates, create dedicated template functions:

```python
from typing import List, Optional
from pydantic import BaseModel
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class ProductReview(BaseModel):
    product_name: str
    rating: int
    pros: List[str]
    cons: List[str]
    summary: str

def create_review_extraction_prompt(
    review_text: str,
    product_category: str,
    include_sentiment: bool = False
) -> str:
    sentiment_instruction = """
    Also include a brief sentiment analysis of the review.
    """ if include_sentiment else ""

    return f"""
    Extract product review information from the following {product_category} review:

    {review_text}

    Please identify:
    - The name of the product being reviewed
    - The numerical rating (1-5)
    - A list of pros/positive points
    - A list of cons/negative points
    - A brief summary of the review
    {sentiment_instruction}
    """

# Use the template function
review_text = """
I recently purchased the UltraSound X300 headphones, and I'm mostly satisfied.
The sound quality is amazing and the battery lasts for days. They're also very
comfortable to wear for long periods. However, they're a bit pricey at $299, and
the Bluetooth occasionally disconnects. Overall, I'd give them 4 out of 5 stars.
"""

prompt = create_review_extraction_prompt(
    review_text=review_text,
    product_category="headphone",
    include_sentiment=True
)

review = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    response_model=ProductReview
)
```

## Best Practices for Prompt Templates

1. **Be explicit about the output format**: Clearly specify what fields you need and in what format
2. **Use consistent language**: Maintain consistent terminology throughout the template
3. **Keep it concise**: Avoid unnecessary verbosity that could confuse the model
4. **Parameterize only what varies**: Only make template parameters for parts that need to change
5. **Include examples for complex tasks**: Provide few-shot examples for more complex extractions
6. **Test with different inputs**: Ensure your template works well with a variety of inputs

## Related Resources

- [Simple Object Extraction](./simple_object.md) - Extracting basic objects
- [List Extraction](./list_extraction.md) - Working with lists of objects
- [Optional Fields](./optional_fields.md) - Handling optional data
- [Prompting](../../concepts/prompting.md) - General prompting concepts
- [Templating](../../concepts/templating.md) - Advanced template techniques

## Next Steps

- Explore [Field Validation](./field_validation.md) for ensuring data quality
- Try [List Extraction](./list_extraction.md) for extracting multiple items
- Learn about [Nested Structure](./nested_structure.md) for complex data