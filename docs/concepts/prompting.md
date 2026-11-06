---
title: Prompt Engineering Best Practices
description: Learn prompt engineering tips for using Pydantic and Instructor effectively.
---

# General Tips for Prompt Engineering

When using Instructor and Pydantic, make your models self-descriptive, modular, and flexible while keeping data integrity.

- Modularity: Design self-contained components for reuse
- Self-description: Use Pydantic's `Field` for clear field descriptions
- Optionality: Use Python's `Optional` type for nullable fields and set defaults
- Standardization: Use enumerations for fields with fixed values; include a fallback option
- Dynamic data: Use key-value pairs for arbitrary properties and limit list lengths
- Entity relationships: Define explicit identifiers and relationship fields
- Contextual logic: Optionally add a "chain of thought" field in reusable components for extra context

## Modular Chain of Thought {#chain-of-thought}

Use chain of thought to improve data quality. You can add it to specific components rather than making it global.

```python hl_lines="4 5"
from pydantic import BaseModel, Field


class Role(BaseModel):
    chain_of_thought: str = Field(
        ..., description="Think step by step to determine the correct title"
    )
    title: str


class UserDetail(BaseModel):
    age: int
    name: str
    role: Role
```

## Utilize Optional Attributes

Use Python's Optional type and set a default value to prevent undesired defaults like empty strings.

```python hl_lines="6"
from typing import Optional
from pydantic import BaseModel, Field


class UserDetail(BaseModel):
    age: int
    name: str
    role: Optional[str] = Field(default=None)
```

## Handling Errors Within Function Calls

Create a wrapper class to hold either the result of an operation or an error message. This lets you stay within a function call even if an error occurs, improving error handling without breaking the code flow.

```python
from pydantic import BaseModel, Field
from typing import Optional


class UserDetail(BaseModel):
    age: int
    name: str
    role: Optional[str] = Field(default=None)


class MaybeUser(BaseModel):
    result: Optional[UserDetail] = Field(default=None)
    error: bool = Field(default=False)
    message: Optional[str]

    def __bool__(self):
        return self.result is not None
```

With the `MaybeUser` class, you can either receive a `UserDetail` object in result or get an error message in message.

### Simplification with the Maybe Pattern

Simplify this using Instructor to create the `Maybe` pattern dynamically from any `BaseModel`.

```python
import instructor
from pydantic import BaseModel


class UserDetail(BaseModel):
    age: int
    name: str


MaybeUser = instructor.Maybe(UserDetail)
```

This lets you quickly create a Maybe type for any class.

## Tips for Enumerations

Use Enums for standardized fields to prevent data misalignment. Always include an "Other" option as a fallback so the model can signal uncertainty.

```python hl_lines="7 12"
from enum import Enum, auto
from pydantic import BaseModel, Field


class Role(Enum):
    PRINCIPAL = auto()
    TEACHER = auto()
    STUDENT = auto()
    OTHER = auto()


class UserDetail(BaseModel):
    age: int
    name: str
    role: Role = Field(
        description="Correctly assign one of the predefined roles to the user."
    )
```

## Literals {#literals}

If you're having a hard time with `Enum` an alternative is to use `Literal`

```python hl_lines="4"
from typing import Literal
from pydantic import BaseModel


class UserDetail(BaseModel):
    age: int
    name: str
    role: Literal["PRINCIPAL", "TEACHER", "STUDENT", "OTHER"]
```

If you'd like to improve performance more you can reiterate the requirements in the field descriptions or in the docstrings.

## Reiterate Long Instructions

For complex attributes, repeat the instructions in the field's description.

```python hl_lines="5 11"
from pydantic import BaseModel, Field


class Role(BaseModel):
    """
    Extract the role based on the following rules ...
    """

    instructions: str = Field(
        ...,
        description="Restate the instructions and rules to correctly determine the title.",
    )
    title: str


class UserDetail(BaseModel):
    age: int
    name: str
    role: Role
```

## Handle Arbitrary Properties

When you need to extract undefined attributes, use a list of key-value pairs.

```python hl_lines="10"
from typing import List
from pydantic import BaseModel, Field


class Property(BaseModel):
    key: str
    value: str


class UserDetail(BaseModel):
    age: int
    name: str
    properties: List[Property] = Field(
        ..., description="Extract any other properties that might be relevant."
    )
```

## Limiting the Length of Lists

When dealing with lists of attributes, especially arbitrary properties, manage the length. Use prompting and enumeration to limit the list length and keep a manageable set of properties.

```python hl_lines="2 9"
from typing import List
from pydantic import BaseModel, Field


class Property(BaseModel):
    index: str = Field(..., description="Monotonically increasing ID")
    key: str
    value: str


class UserDetail(BaseModel):
    age: int
    name: str
    properties: List[Property] = Field(
        ...,
        description="Numbered list of arbitrary extracted properties, should be less than 6",
    )
```

### Using Tuples for Simple Types

For simple types, tuples can be a more compact alternative to custom classes, especially when the properties don't require additional descriptions.

```python hl_lines="4"
from typing import List, Tuple
from pydantic import BaseModel, Field


class UserDetail(BaseModel):
    age: int
    name: str
    properties: List[Tuple[int, str]] = Field(
        ...,
        description="Numbered list of arbitrary extracted properties, should be less than 6",
    )
```

## Advanced Arbitrary Properties

For multiple users, use consistent key names when extracting properties.

```python
from typing import List
from pydantic import BaseModel


class UserDetail(BaseModel):
    id: int
    age: int
    name: str


class UserDetails(BaseModel):
    """
    Extract information for multiple users.
    Use consistent key names for properties across users.
    """

    users: List[UserDetail]
```

This refined guide should offer a cleaner and more organized approach to structure engineering in Python.

## Defining Relationships Between Entities

When relationships exist between entities, define them explicitly in the model. The following example shows how to define relationships between users by adding an id and a friends field:

```python hl_lines="2 5 8"
from typing import List
from pydantic import BaseModel, Field


class UserDetail(BaseModel):
    id: int = Field(..., description="Unique identifier for each user.")
    age: int
    name: str
    friends: List[int] = Field(
        ...,
        description="Correct and complete list of friend IDs, representing relationships between users.",
    )


class UserRelationships(BaseModel):
    users: List[UserDetail] = Field(
        ...,
        description="Collection of users, correctly capturing the relationships among them.",
    )
```

## Reusing Components with Different Contexts

You can reuse the same component for different contexts within a model. In this example, the TimeRange component is used for both work_time and leisure_time.

```python hl_lines="9 10"
from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    start_time: int = Field(..., description="The start time in hours.")
    end_time: int = Field(..., description="The end time in hours.")


class UserDetail(BaseModel):
    id: int = Field(..., description="Unique identifier for each user.")
    age: int
    name: str
    work_time: TimeRange = Field(
        ..., description="Time range during which the user is working."
    )
    leisure_time: TimeRange = Field(
        ..., description="Time range reserved for leisure activities."
    )
```

Sometimes, a component like TimeRange may need context or additional logic to work well. Adding a "chain of thought" field within the component can help understand or optimize the time range allocations.

```python hl_lines="2"
from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    chain_of_thought: str = Field(
        ..., description="Step by step reasoning to get the correct time range"
    )
    start_time: int = Field(..., description="The start time in hours.")
    end_time: int = Field(..., description="The end time in hours.")
```
