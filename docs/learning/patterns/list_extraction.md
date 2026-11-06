---
title: List Extraction from LLMs Tutorial
description: Master extracting multiple structured objects from language models using Instructor with type-safe list validation.
---

# List Extraction Tutorial: Extract Multiple Objects from LLMs

Master the art of extracting lists and arrays from LLMs in this comprehensive tutorial. Learn how to use Instructor to extract multiple structured objects from language models like GPT-4, Claude, and Gemini with type-safe validation.

## Basic List Extraction

To extract a list of items, you define a model for a single item and then use Python's typing system to specify you want a list of that type:

```python
from typing import List
from pydantic import BaseModel, Field
import instructor
# Initialize the client
client = instructor.from_provider("openai/gpt-5-nano")

# Define a single item model
class Person(BaseModel):
    name: str = Field(..., description="The person's full name")
    age: int = Field(..., description="The person's age in years")

# Define a wrapper model for the list
class PeopleList(BaseModel):
    people: List[Person] = Field(..., description="List of people mentioned in the text")

# Extract the list
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": """
        Here's information about some people:
        - John Smith is 35 years old
        - Mary Johnson is 28 years old
        - Robert Davis is 42 years old
        """}
    ],
    response_model=PeopleList
)

# Access the extracted data
for i, person in enumerate(response.people):
    print(f"Person {i+1}: {person.name}, {person.age} years old")
```

This example shows how to:
1. Define a model for a single item (`Person`)
2. Create a wrapper model that contains a list of items (`PeopleList`)
3. Access each item in the list through the response

## Direct List Extraction

You can also extract a list directly without a wrapper model:

```python
from typing import List
from pydantic import BaseModel, Field
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class Book(BaseModel):
    title: str
    author: str
    publication_year: int

# Extract a list directly
books = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": """
        Classic novels:
        1. To Kill a Mockingbird by Harper Lee (1960)
        2. 1984 by George Orwell (1949)
        3. The Great Gatsby by F. Scott Fitzgerald (1925)
        """}
    ],
    response_model=List[Book]  # Direct list extraction
)

# Access the extracted data
for book in books:
    print(f"{book.title} by {book.author} ({book.publication_year})")
```

## Nested Lists

You can extract nested lists by combining list types:

```python
from typing import List
from pydantic import BaseModel, Field
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class Author(BaseModel):
    name: str
    nationality: str

class Book(BaseModel):
    title: str
    authors: List[Author]  # Nested list of authors
    publication_year: int

# Extract data with nested lists
books = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": """
        Book 1: "Good Omens" (1990)
        Authors: Terry Pratchett (British), Neil Gaiman (British)

        Book 2: "The Talisman" (1984)
        Authors: Stephen King (American), Peter Straub (American)
        """}
    ],
    response_model=List[Book]
)

# Access the nested data
for book in books:
    author_names = ", ".join([author.name for author in book.authors])
    print(f"{book.title} ({book.publication_year}) by {author_names}")
```

## Using Streaming with Lists

You can stream list extraction results using Instructor's streaming capabilities:

```python
from typing import List
import instructor
from pydantic import BaseModel, Field
client = instructor.from_provider("openai/gpt-5-nano")

class Task(BaseModel):
    description: str
    priority: str
    deadline: str

# Stream a list of tasks
for task in client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "Generate a list of 5 sample tasks for a project manager"}
    ],
    response_model=List[Task],
    stream=True
):
    print(f"Received task: {task.description} (Priority: {task.priority}, Deadline: {task.deadline})")
```

For more information on streaming, see the [Streaming Basics](../streaming/basics.md) and [Streaming Lists](../streaming/lists.md) guides.

## List Validation

You can add validation for both individual items and the entire list:

```python
from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class Product(BaseModel):
    name: str
    price: float

    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError("Price must be greater than zero")
        return v

class ProductList(BaseModel):
    products: List[Product] = Field(..., min_items=1)

    @model_validator(mode='after')
    def validate_unique_names(self):
        names = [p.name for p in self.products]
        if len(names) != len(set(names)):
            raise ValueError("All product names must be unique")
        return self

# Extract list with validation
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "List of products: Headphones ($50), Speakers ($80), Earbuds ($30)"}
    ],
    response_model=ProductList
)
```

For more on validation, see [Field Validation](./field_validation.md) and [Validation Basics](../validation/basics.md).

## List Constraints

You can add constraints to lists using Pydantic's Field:

```python
from typing import List
from pydantic import BaseModel, Field
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class Ingredient(BaseModel):
    name: str
    amount: str

class Recipe(BaseModel):
    title: str
    ingredients: List[Ingredient] = Field(
        ...,
        min_items=2,         # Minimum 2 ingredients
        max_items=10,        # Maximum 10 ingredients
        description="List of ingredients needed for the recipe"
    )
    steps: List[str] = Field(
        ...,
        min_items=1,
        description="Step-by-step instructions to prepare the recipe"
    )
```

## Real-world Example: Task Extraction

Here's a more complete example for extracting a list of tasks from a meeting transcript:

```python
from typing import List, Optional
from pydantic import BaseModel, Field
import instructor
from datetime import date
client = instructor.from_provider("openai/gpt-5-nano")

class Assignee(BaseModel):
    name: str
    email: Optional[str] = None

class ActionItem(BaseModel):
    description: str = Field(..., description="The task that needs to be completed")
    assignee: Assignee = Field(..., description="The person responsible for the task")
    due_date: Optional[date] = Field(None, description="The deadline for the task")
    priority: str = Field(..., description="Priority level: Low, Medium, or High")

# Extract action items from meeting notes
action_items = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": """
        Meeting Notes - Project Kickoff
        Date: 2023-05-15

        Attendees: John (john@example.com), Sarah (sarah@example.com), Mike

        Discussion points:
        1. John will prepare the project timeline by next Friday. This is high priority.
        2. Sarah needs to contact the client for requirements clarification by Wednesday. Medium priority.
        3. Mike is responsible for setting up the development environment. Due by tomorrow, high priority.
        """}
    ],
    response_model=List[ActionItem]
)

# Process the extracted action items
for item in action_items:
    due_str = item.due_date.isoformat() if item.due_date else "Not specified"
    print(f"Task: {item.description}")
    print(f"Assignee: {item.assignee.name} ({item.assignee.email or 'No email'})")
    print(f"Due: {due_str}, Priority: {item.priority}")
    print("---")
```

For a more detailed example, see the [Action Items Extraction](../../examples/action_items.md) example.

## Related Resources

- [Simple Object Extraction](./simple_object.md) - Extracting single objects
- [Nested Structure](./nested_structure.md) - Working with complex nested data
- [Streaming Lists](../streaming/lists.md) - Streaming list results
- [Lists and Arrays](../../concepts/lists.md) - Concepts related to list extraction

## Next Steps

- Learn about [Nested Structure](./nested_structure.md) for complex data
- Explore [Streaming Lists](../streaming/lists.md) for handling large lists
- Check out [Field Validation](./field_validation.md) for validation techniques