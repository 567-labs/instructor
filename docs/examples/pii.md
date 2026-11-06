---
title: Extracting and Scrubbing PII Data with OpenAI
description: Learn to extract and sanitize Personally Identifiable Information (PII) from documents using OpenAI's ChatCompletion model and Python.
---

# PII Data Extraction and Scrubbing

## Overview

This example demonstrates the usage of OpenAI's ChatCompletion model for the extraction and scrubbing of Personally Identifiable Information (PII) from a document. The code defines Pydantic models to manage the PII data and offers methods for both extraction and sanitation.

## Defining the Structures

First, Pydantic models are defined to represent the PII data and the overall structure for PII data extraction.

```python
from typing import List
from pydantic import BaseModel


# Define Schemas for PII data
class Data(BaseModel):
    index: int
    data_type: str
    pii_value: str


class PIIDataExtraction(BaseModel):
    """
    Extracted PII data from a document, all data_types should try to have consistent property names
    """

    private_data: List[Data]

    def scrub_data(self, content: str) -> str:
        """
        Iterates over the private data and replaces the value with a placeholder in the form of
        <{data_type}_{i}>
        """
        for i, data in enumerate(self.private_data):
            content = content.replace(data.pii_value, f"<{data.data_type}_{i}>")
        return content
```

## Extracting PII Data

The OpenAI API is utilized to extract PII information from a given document.

```python
import instructor

# <%hide%>
from typing import List
from pydantic import BaseModel


# Define Schemas for PII data
class Data(BaseModel):
    index: int
    data_type: str
    pii_value: str


class PIIDataExtraction(BaseModel):
    """
    Extracted PII data from a document, all data_types should try to have consistent property names
    """

    private_data: List[Data]

    def scrub_data(self, content: str) -> str:
        """
        Iterates over the private data and replaces the value with a placeholder in the form of
        <{data_type}_{i}>
        """
        for i, data in enumerate(self.private_data):
            content = content.replace(data.pii_value, f"<{data.data_type}_{i}>")
        return content


# <%hide%>

client = instructor.from_provider("openai/gpt-5-nano")

EXAMPLE_DOCUMENT = """
# Fake Document with PII for Testing PII Scrubbing Model
# (The content here)
"""

pii_data = client.create(
    model="gpt-4o-mini",
    response_model=PIIDataExtraction,
    messages=[
        {
            "role": "system",
            "content": "You are a world class PII scrubbing model, Extract the PII data from the following document",
        },
        {
            "role": "user",
            "content": EXAMPLE_DOCUMENT,
        },
    ],
)  # type: ignore

print("Extracted PII Data:")
#> Extracted PII Data:
print(pii_data.model_dump_json())
"""
{"private_data":[{"index":1,"data_type":"Name","pii_value":"John Doe"},{"index":2,"data_type":"Email","pii_value":"john.doe@example.com"},{"index":3,"data_type":"Phone","pii_value":"+1234567890"},{"index":4,"data_type":"Address","pii_value":"1234 Elm Street, Springfield, IL 62704"},{"index":5,"data_type":"SSN","pii_value":"123-45-6789"}]}
"""
```

### Output of Extracted PII Data

```json
{
  "private_data": [
    {
      "index": 0,
      "data_type": "date",
      "pii_value": "01/02/1980"
    },
    {
      "index": 1,
      "data_type": "ssn",
      "pii_value": "123-45-6789"
    },
    {
      "index": 2,
      "data_type": "email",
      "pii_value": "john.doe@email.com"
    },
    {
      "index": 3,
      "data_type": "phone",
      "pii_value": "555-123-4567"
    },
    {
      "index": 4,
      "data_type": "address",
      "pii_value": "123 Main St, Springfield, IL, 62704"
    }
  ]
}
```

## Scrubbing PII Data

After extracting the PII data, the `scrub_data` method is used to sanitize the document.

```python
# <%hide%>
from typing import List
from pydantic import BaseModel


# Define Schemas for PII data
class Data(BaseModel):
    index: int
    data_type: str
    pii_value: str


class PIIDataExtraction(BaseModel):
    """
    Extracted PII data from a document, all data_types should try to have consistent property names
    """

    private_data: List[Data]

    def scrub_data(self, content: str) -> str:
        """
        Iterates over the private data and replaces the value with a placeholder in the form of
        <{data_type}_{i}>
        """
        for i, data in enumerate(self.private_data):
            content = content.replace(data.pii_value, f"<{data.data_type}_{i}>")
        return content


pii_data = PIIDataExtraction(
    private_data=[
        {"index": 0, "data_type": "date", "pii_value": "01/02/1980"},
        {"index": 1, "data_type": "ssn", "pii_value": "123-45-6789"},
        {"index": 2, "data_type": "email", "pii_value": "john.doe@email.com"},
        {"index": 3, "data_type": "phone", "pii_value": "555-123-4567"},
        {
            "index": 4,
            "data_type": "address",
            "pii_value": "123 Main St, Springfield, IL, 62704",
        },
    ]
)

EXAMPLE_DOCUMENT = """
# Fake Document with PII for Testing PII Scrubbing Model
# He was born on 01/02/1980. His social security number is 123-45-6789. He has been using the email address john.doe@email.com for years, and he can always be reached at 555-123-4567.
"""
# <%hide%>
print("Scrubbed Document:")
#> Scrubbed Document:
print(pii_data.scrub_data(EXAMPLE_DOCUMENT))
"""
# Fake Document with PII for Testing PII Scrubbing Model
# He was born on <date_0>. His social security number is <ssn_1>. He has been using the email address <email_2> for years, and he can always be reached at <phone_3>.
"""
```

### Output of Scrubbed Document

```plaintext
# Fake Document with PII for Testing PII Scrubbing Model

## Personal Story

John Doe was born on <date_0>. His social security number is <ssn_1>. He has been using the email address <email_2> for years, and he can always be reached at <phone_3>.

## Residence

John currently resides at <address_4>. He's been living there for about 5 years now.
```
