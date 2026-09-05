---
authors:
- jxnl
categories:
- LLM Techniques
comments: true
date: 2024-09-19
description: Explore the integration of Jinja templating in the Instructor for enhanced
  formatting, validation, versioning, and secure logging.
draft: false
tags:
- Jinja
- Templating
- Pydantic
- API Development
- Data Validation
---

# Instructor Proposal: Integrating Jinja Templating

As the creator of Instructor, I've always aimed to keep our product development streamlined and avoid unnecessary complexity. However, I'm now convinced that it's time to incorporate better templating into our data structure, specifically by integrating Jinja.

This decision serves multiple purposes:

1. It addresses the growing complexity in my prompt formatting needs
2. It allows us to differentiate ourselves from the standard library while adding proven utility.
3. It aligns with the practices I've consistently employed in both production and client code.
4. It provides an opportunity to introduce API changes that have been tested in private versions of Instructor.

## Why Jinja is the Right Choice

1. **Formatting Capabilities**
   - Prompt formatting complexity has increased.
   - List iteration and conditional implementation are necessary for formatting.
   - This improves chunk generation, few shots, and dynamic rules.

2. **Validation**
   - Jinja template variables serve rendering and validation purposes.
   - Pydantic's validation context allows access to template variables in validation functions.

3. **Versioning and Logging**
   - Render variable separation enhances prompt versioning and logging.
   - Template variable diffing simplifies prompt change comparisons.

By integrating Jinja into Instructor, we're not just adding a feature; we're enhancing our ability to handle complex formatting, improve validation processes, and streamline our versioning and logging capabilities. This addition will significantly boost the power and flexibility of Instructor, making it an even more robust tool for our users.

## Enhancing Formatting Capabilities

In Instructor, we propose implementing a new `context` keyword in our create methods. This addition will allow users to render the prompt using a provided context, leveraging Jinja's templating capabilities. Here's how it would work:

1. Users pass a `context` dictionary to the create method.
2. The prompt template, written in Jinja syntax, is defined in the `content` field of the message.
3. Instructor renders the prompt using the provided context, filling in the template variables.

This approach offers these benefits:

- Separation of prompt structure and dynamic content
- Management of complex prompts with conditionals and loops
- Reusability of prompt templates across different contexts

Let's look at an example to illustrate this feature:

```python
client.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": """
                You are a {{ role }} tasks with the following question

                <question>
                {{ question }}
                </question>

                Use the following context to answer the question, make sure to return [id] for every citation:

                <context>
                {% for chunk in context %}
                  <context_chunk>
                    <id>{{ chunk.id }}</id>
                    <text>{{ chunk.text }}</text>
                  </context_chunk>
                {% endfor %}
                </context>

                {% if rules %}
                Make sure to follow these rules:

                {% for rule in rules %}
                  * {{ rule }}
                {% endfor %}
                {% endif %}
            """,
        },
    ],
    context={
        "role": "professional educator",
        "question": "What is the capital of France?",
        "context": [
            {"id": 1, "text": "Paris is the capital of France."},
            {"id": 2, "text": "France is a country in Europe."},
        ],
        "rules": ["Use markdown."],
    },
)
```

## Validation

Let's consider a scenario where we redact words from text. By using `ValidationInfo` to access context and passing it to the validator and template, we can implement a system for handling sensitive information. This approach allows us to:

1. Validate input to ensure it doesn't contain banned words.
2. Redact patterns using regular expressions.
3. Provide instructions to the language model about word usage restrictions.

Here's an example demonstrating this concept using Pydantic validators:

```python
from pydantic import BaseModel, ValidationInfo, field_validator


class Response(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def no_banned_words(cls, v: str, info: ValidationInfo):
        context = info.context
        if context:
            banned_words = context.get('banned_words', set())
            banned_words_found = [
                word for word in banned_words if word.lower() in v.lower()
            ]
            if banned_words_found:
                raise ValueError(
                    f"Banned words found in text: {', '.join(banned_words_found)}, rewrite it but just without the banned words"
                )
        return v

    @field_validator('text')
    @classmethod
    def redact_regex(cls, v: str, info: ValidationInfo):
        context = info.context
        if context:
            redact_patterns = context.get('redact_patterns', [])
            for pattern in redact_patterns:
                v = re.sub(pattern, '****', v)
        return v


response = client.create(
    model="gpt-4o",
    response_model=Response,
    messages=[
        {
            "role": "user",
            "content": """
                Write about a {{ topic }}

                {% if banned_words %}
                You must not use the following banned words:

                <banned_words>
                {% for word in banned_words %}
                * {{ word }}
                {% endfor %}
                </banned_words>
                {% endif %}
              """,
        },
    ],
    context={
        "topic": "jason and now his phone number is 123-456-7890",
        "banned_words": ["jason"],
        "redact_patterns": [
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone number pattern
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
        ],
    },
    max_retries=3,
)

print(response.text)
#> While i can't say his name anymore, his phone number is ****
```

## Better Versioning and Logging

With the separation of prompt templates and variables, we gain several advantages:

1. Version Control: We can now version the templates and retrieve the appropriate one for a given prompt. This allows for better management of template history, diffing and comparison.

2. Enhanced Logging: The separation facilitates structured logging, enabling easier debugging and integration with various logging sinks, databases, and observability tools like OpenTelemetry.

3. Security: Sensitive information in variables can be handled separately from the templates, allowing for better access control and data protection.

This separation of concerns adheres to best practices in software design, resulting in a more maintainable, scalable, and robust system for managing prompts and their associated data.

### Side effect of Context also being Pydantic Models

Since they are just python objects we can use Pydantic models to validate the context and also control how they are rendered, so even secret information can be dynamically rendered!
Consider using secret string to pass in sensitive information to the llm.

```python
from pydantic import BaseModel, SecretStr


class UserContext(BaseModel):
    name: str
    address: SecretStr


class Address(BaseModel):
    street: SecretStr
    city: str
    state: str
    zipcode: str


def normalize_address(address: Address):
    context = UserContext(username="scolvin", address=address)
    address = client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "{{ user.name }} is `{{ user.address.get_secret_value() }}`, normalize it to an address object",
            },
        ],
        context={"user": context},
    )
    print(context)
    #> UserContext(username='jliu', address="******")
    print(address)
    #> Address(street='******', city="Toronto", state="Ontario", zipcode="M5A 0J3")
    logger.info(
        f"Normalized address: {address}",
        extra={"user_context": context, "address": address},
    )
    return address
```

This approach offers several advantages:

1. Secure logging: You can confidently log your template variables without risking the exposure of sensitive information.
2. Type safety: Pydantic models provide type checking and validation, reducing the risk of errors.
3. Flexibility: You can easily control how different types of data are displayed or used in templates.