---
title: Using Pydantic Models for Structured Outputs
description: Learn how to define LLM output schemas with Pydantic models.
---

# Response Model

Define LLM output schemas using `pydantic.BaseModel`. For more details, see the [Pydantic documentation](https://docs.pydantic.dev/latest/concepts/models/).

After defining a Pydantic model, use it as the `response_model` in your client `create` calls. The `response_model` parameter:

- Defines the schema and prompts for the language model
- Validates the response from the API
- Returns a Pydantic model instance

## Prompting

Use docstrings and field annotations to define the prompt for generating responses.

```python
from pydantic import BaseModel, Field
import instructor


class User(BaseModel):
    """
    This is the prompt that will be used to generate the response.
    Any instructions here will be passed to the language model.
    """

    name: str = Field(description="The name of the user.")
    age: int = Field(description="The age of the user.")


client = instructor.from_provider("openai/gpt-4o-mini")

user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30 years old"}],
)
```

Docstrings, types, and field annotations are used to generate the prompt. The `create` method uses this prompt to generate the response.

## Optional Values

Use `Optional` and `default` to make fields optional when sent to the language model.

```python
from pydantic import BaseModel, Field
from typing import Optional
import instructor


class User(BaseModel):
    name: str = Field(description="The name of the user.")
    age: int = Field(description="The age of the user.")
    email: Optional[str] = Field(description="The email of the user.", default=None)


client = instructor.from_provider("openai/gpt-4o-mini")

user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30 years old"}],
)
```

Fields can also be omitted from the schema sent to the language model using Pydantic's `SkipJsonSchema` annotation. See [Fields](fields.md#omitting-fields-from-schema-sent-to-the-language-model) for details.

## Dynamic Model Creation

Create models at runtime using Pydantic's `create_model` function:

```python
from pydantic import BaseModel, create_model


class FooModel(BaseModel):
    foo: str
    bar: int = 123


BarModel = create_model(
    'BarModel',
    apple=(str, 'russet'),
    banana=(str, 'yellow'),
    __base__=FooModel,
)
print(BarModel)
#> <class '__main__.BarModel'>
print(BarModel.model_fields.keys())
#> dict_keys(['foo', 'bar', 'apple', 'banana'])
```

??? notes "When would I use this?"

    Consider a situation where the model is dynamically defined, based on some configuration or database. For example, we could have a database table that stores the properties of a model for
    some model name or id. We could then query the database for the properties of the model and use that to create the model.

    ```sql
    SELECT property_name, property_type, description
    FROM prompt
    WHERE model_name = {model_name}
    ```

    We can then use this information to create the model.

    ```python
    from pydantic import BaseModel, create_model, Field
    from typing import List

    types = {
        'string': str,
        'integer': int,
        'boolean': bool,
        'number': float,
        'List[str]': List[str],
    }

    # Mocked cursor.fetchall()
    cursor = [
        ('name', 'string', 'The name of the user.'),
        ('age', 'integer', 'The age of the user.'),
        ('email', 'string', 'The email of the user.'),
    ]

    BarModel = create_model(
        'User',
        **{
            property_name: (types[property_type], Field(description=description))
            for property_name, property_type, description in cursor
        },
        __base__=BaseModel,
    )

    print(BarModel.model_json_schema())
    """
    {
        'properties': {
            'name': {
                'description': 'The name of the user.',
                'title': 'Name',
                'type': 'string',
            },
            'age': {
                'description': 'The age of the user.',
                'title': 'Age',
                'type': 'integer',
            },
            'email': {
                'description': 'The email of the user.',
                'title': 'Email',
                'type': 'string',
            },
        },
        'required': ['name', 'age', 'email'],
        'title': 'User',
        'type': 'object',
    }
    """
    ```

    This would be useful when different users have different descriptions for the same model. We can use the same model but have different prompts for each user.

## Adding Behavior

Add methods to Pydantic models like any Python class. This lets you add custom logic to your models.

```python
from pydantic import BaseModel
from typing import Literal

import instructor

client = instructor.from_provider("openai/gpt-4.1-mini")


class SearchQuery(BaseModel):
    query: str
    query_type: Literal["web", "image", "video"]

    def execute(self):
        print(f"Searching for {self.query} of type {self.query_type}")
        #> Searching for cat of type image
        return "Results for cat"


query = client.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Search for a picture of a cat"}],
    response_model=SearchQuery,
)

results = query.execute()
print(results)
#> Results for cat
```

Now we can call `execute` on our model instance after extracting it from a language model. If you want to see more examples of this checkout our post on [RAG is more than embeddings](../blog/posts/rag-and-beyond.md)

## See Also

- [Response Models Tutorial](../learning/getting_started/response_models.md) - Step-by-step guide to creating response models
- [Simple Object Extraction](../learning/patterns/simple_object.md) - Basic extraction patterns
- [Nested Structures](../learning/patterns/nested_structure.md) - Complex hierarchical models
- [Optional Fields](../learning/patterns/optional_fields.md) - Working with optional data
- [Types](./types.md) - Working with different data types
- [Fields](./fields.md) - Advanced field configuration
