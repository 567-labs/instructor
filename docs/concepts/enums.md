---
title: Using Enums and Literals in Pydantic for Role Management
description: Learn how to implement Enums and Literals in Pydantic to manage standardized user roles with a fallback option.
---

To prevent data misalignment, we can use Enums for standardized fields. Always include an "Other" option as a fallback so the model can signal uncertainty.

```python hl_lines="7 12"
from pydantic import BaseModel, Field
from enum import Enum


class Role(Enum):
    PRINCIPAL = "PRINCIPAL"
    TEACHER = "TEACHER"
    STUDENT = "STUDENT"
    OTHER = "OTHER"


class UserDetail(BaseModel):
    age: int
    name: str
    role: Role = Field(
        description="Correctly assign one of the predefined roles to the user."
    )
```

If you're having a hard time with `Enum` an alternative is to use `Literal` instead.

```python hl_lines="4"
from typing import Literal
from pydantic import BaseModel


class UserDetail(BaseModel):
    age: int
    name: str
    role: Literal["PRINCIPAL", "TEACHER", "STUDENT", "OTHER"]
```

## See Also

- [Types](./types.md) - Working with different data types including Literal
- [Union Types](./unions.md) - Using unions with enums for multiple choices
- [Response Models](./models.md) - Using enums in Pydantic models
- [Fields](./fields.md) - Customizing enum fields with Field metadata
