---
title: Groq AI Integration - Fast Structured Outputs
description: Use Groq AI with Instructor for fast structured outputs. Leverage Groq's high-speed inference for real-time structured data extraction.
---

# Structured Outputs using Groq
Instead of using Openai or Antrophic you can now also use groq for inference by using from_groq.

The examples are using mixtral-8x7b model.

## GroqCloud API
To use groq you need to obtain a groq API key.
Go to [groqcloud](https://console.groq.com) and login. Select API Keys from the left menu and then select Create API key to create a new key.

## Use example
Some pip packages need to be installed to use the example:
```
pip install instructor groq pydantic openai anthropic
```
You need to export the groq API key:
```
export GROQ_API_KEY=<your-api-key>
```

An example:
```python
from pydantic import BaseModel, Field
from typing import List
import instructor


class Character(BaseModel):
    name: str
    fact: List[str] = Field(..., description="A list of facts about the subject")


# Use from_provider for simplified setup
client = instructor.from_provider("groq/mixtral-8x7b-32768", mode=instructor.Mode.TOOLS)

resp = client.create(
    model="mixtral-8x7b-32768",
    messages=[
        {
            "role": "user",
            "content": "Tell me about the company Tesla",
        }
    ],
    response_model=Character,
)
print(resp.model_dump_json(indent=2))
"""
{
  "name": "Tesla",
  "fact": [
    "electric vehicle manufacturer",
    "solar panel producer",
    "based in Palo Alto, California",
    "founded in 2003 by Elon Musk"
  ]
}
"""
```
You can find another example called groq_example2.py under examples/groq of this repository.
