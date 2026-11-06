---
title: Nested Structure Extraction with Instructor
description: Learn how to extract complex nested data structures from LLMs using hierarchical Pydantic models.
---

# Simple Nested Structure

This guide explains how to extract nested structured data using Instructor. Nested structures allow you to represent complex, hierarchical data relationships.

## Understanding Nested Structures

Nested structures are objects that contain other objects as fields. They're useful for representing:

1. Parent-child relationships
2. Complex entities with sub-components
3. Hierarchical data
4. Related data that belongs together

## Basic Nested Structure Example

Here's a simple example of extracting a nested structure:

```python
from pydantic import BaseModel, Field
import instructor
from typing import List, Optional
# Initialize the client
client = instructor.from_provider("openai/gpt-5-nano")

# Define nested models
class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: int
    address: Address  # Nested structure

# Extract the nested data
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": """
        John Smith is 35 years old.
        He lives at 123 Main Street, Boston, MA 02108.
        """}
    ],
    response_model=Person
)

# Access the nested data
print(f"Name: {response.name}")
print(f"Age: {response.age}")
print(f"Address: {response.address.street}, {response.address.city}, "
      f"{response.address.state} {response.address.zip_code}")
```

## Multiple Levels of Nesting

You can use multiple levels of nesting for more complex structures:

```python
from pydantic import BaseModel, Field
import instructor
from typing import List, Optional
client = instructor.from_provider("openai/gpt-5-nano")

class EmployeeDetails(BaseModel):
    department: str
    position: str
    start_date: str

class ContactInfo(BaseModel):
    phone: str
    email: str

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: int
    contact: ContactInfo  # First level nesting
    address: Address      # First level nesting
    employment: Optional[EmployeeDetails] = None  # Optional nested structure

# Extract deeply nested data
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": """
        Employee Profile:
        Name: Jane Doe
        Age: 32
        Phone: (555) 123-4567
        Email: jane.doe@example.com
        Address: 456 Oak Avenue, Chicago, IL 60601
        Department: Engineering
        Position: Senior Developer
        Start Date: 2021-03-15
        """}
    ],
    response_model=Person
)
```

## Nested Lists

You can combine nesting with lists to represent complex collections:

```python
from pydantic import BaseModel, Field
import instructor
from typing import List
client = instructor.from_provider("openai/gpt-5-nano")

class Ingredient(BaseModel):
    name: str
    amount: str
    unit: str

class Recipe(BaseModel):
    title: str
    description: str
    ingredients: List[Ingredient]  # Nested list of ingredients
    steps: List[str]  # List of strings

# Extract nested list data
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": """
        Recipe: Chocolate Chip Cookies

        Description: Classic homemade chocolate chip cookies that are soft in the middle and crispy on the edges.

        Ingredients:
        - 2 1/4 cups all-purpose flour
        - 1 teaspoon baking soda
        - 1 teaspoon salt
        - 1 cup butter
        - 3/4 cup white sugar
        - 3/4 cup brown sugar
        - 2 eggs
        - 2 teaspoons vanilla extract
        - 2 cups chocolate chips

        Instructions:
        1. Preheat oven to 375°F (190°C)
        2. Mix flour, baking soda, and salt
        3. Cream butter and sugars, then add eggs and vanilla
        4. Gradually add dry ingredients
        5. Stir in chocolate chips
        6. Drop by rounded tablespoons onto ungreased baking sheets
        7. Bake for 9 to 11 minutes or until golden brown
        8. Cool on wire racks
        """}
    ],
    response_model=Recipe
)
```

For more information on working with lists, see the [List Extraction](list_extraction.md) guide.

## Handling Optional Nested Fields

Sometimes parts of a nested structure might be missing. Use Optional to handle this:

```python
from pydantic import BaseModel, Field
import instructor
from typing import Optional
client = instructor.from_provider("openai/gpt-5-nano")

class SocialMedia(BaseModel):
    twitter: Optional[str] = None
    linkedin: Optional[str] = None
    instagram: Optional[str] = None

class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None
    social: Optional[SocialMedia] = None  # Optional nested structure

class Person(BaseModel):
    name: str
    contact: ContactInfo
```

For more information on optional fields, see the [Optional Fields](optional_fields.md) guide.

## Nested Structure Validation

You can add validation to nested structures at any level:

```python
from pydantic import BaseModel, Field, field_validator, model_validator
import instructor
import re
client = instructor.from_provider("openai/gpt-5-nano")

class EmailContact(BaseModel):
    email: str

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError("Invalid email format")
        return v

class Customer(BaseModel):
    name: str
    contact: EmailContact  # Nested structure with its own validation

    @model_validator(mode='after')
    def validate_name_email_match(self):
        name_part = self.name.lower().split()[0]
        if name_part not in self.contact.email.lower():
            print(f"Warning: Email {self.contact.email} may not match name {self.name}")
        return self
```

For more on validation, see [Field Validation](field_validation.md) and [Validation Basics](../validation/basics.md).

## Working with Recursive Structures

For more complex hierarchical data, you can use recursive structures:

```python
from typing import List, Optional
from pydantic import BaseModel, Field
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class Comment(BaseModel):
    text: str
    author: str
    replies: List["Comment"] = []  # Recursive structure

# Update the Comment class reference for Pydantic
Comment.model_rebuild()

class Post(BaseModel):
    title: str
    content: str
    author: str
    comments: List[Comment] = []

# Extract recursive nested data
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": """
        Blog Post: "Python Tips and Tricks"
        Author: John Smith
        Content: Here are some helpful Python tips for beginners...

        Comments:
        1. Alice: "Great post! Very helpful."
           - Bob: "I agree, I learned a lot."
             - Alice: "Bob, did you try the last example?"
           - Charlie: "Thanks for sharing this."
        2. David: "Could you explain the second tip more?"
           - John: "Sure, I'll add more details."
        """}
    ],
    response_model=Post
)
```

For more advanced recursive structures, see the [Recursive Structures](../../examples/recursive.md) guide.

## Real-world Example: Organization Structure

Here's a more complete example extracting an organization structure:

```python
from typing import List, Optional
from pydantic import BaseModel, Field
import instructor
client = instructor.from_provider("openai/gpt-5-nano")

class Employee(BaseModel):
    name: str
    title: str

class Department(BaseModel):
    name: str
    head: Employee
    employees: List[Employee]
    sub_departments: List["Department"] = []

# Update for Pydantic's recursive model support
Department.model_rebuild()

class Organization(BaseModel):
    name: str
    ceo: Employee
    departments: List[Department]

# Extract organization structure
response = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": """
        Acme Corporation
        CEO: Jane Smith, Chief Executive Officer

        Departments:

        1. Engineering
           Head: Bob Johnson, CTO
           Employees:
           - Sarah Lee, Senior Engineer
           - Tom Brown, Software Developer

           Sub-departments:
           - Frontend Team
             Head: Lisa Wang, Frontend Lead
             Employees:
             - Mike Chen, UI Developer
             - Ana Garcia, UX Designer

           - Backend Team
             Head: David Kim, Backend Lead
             Employees:
             - James Wright, Database Engineer
             - Rachel Patel, API Developer

        2. Marketing
           Head: Michael Davis, CMO
           Employees:
           - Jennifer Miller, Marketing Specialist
           - Robert Chen, Content Creator
        """}
    ],
    response_model=Organization
)
```


## Related Resources

- [Simple Object Extraction](./simple_object.md) - Extracting basic objects
- [List Extraction](./list_extraction.md) - Working with lists of objects
- [Optional Fields](./optional_fields.md) - Handling optional data
- [Recursive Structures](../../examples/recursive.md) - Building more complex hierarchies
- [Field Validation](./field_validation.md) - Adding validation to your fields

