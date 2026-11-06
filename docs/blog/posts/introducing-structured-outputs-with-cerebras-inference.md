---
authors:
  - ivanleomk
  - sarahchieng
categories:
  - API Development
  - Pydantic
  - Performance Optimization
comments: true
date: 2024-10-15
description:
  Learn how to use Cerebras Inference for structured outputs, faster model
  inference, and seamless integration with Pydantic models.
draft: false
slug: introducing-structured-outputs-with-cerebras-inference
tags:
  - Cerebras Inference
  - Pydantic
  - API Integration
  - Fast Inference
  - Structured Outputs
---

# Introducing structured outputs with Cerebras Inference

## What's Cerebras?

Cerebras offers the fastest inference on the market, 20x faster than on GPUs.

Sign up for a Cerebras Inference API key here at [cloud.cerebras.ai](http://cloud.cerebras.ai).

### Basic Usage

To get guaranteed structured outputs with Cerebras Inference, you

<!-- more -->

1. Create a new Instructor client with the `from_cerebras` method
2. Define a Pydantic model to pass into the `response_model` parameter
3. Get back a validated response exactly as you would expect

You'll also need to install the Cerebras SDK to use the client. You can install it with the command below.

<!-- more -->

```bash
pip install "instructor[cerebras_cloud_sdk]"
```

This ensures that you have the necessary dependencies to use the Cerebras SDK with instructor.

### Getting Started

Before running the following code, you'll need to make sure that you have your CEREBRAS_API_KEY. Sign up for one [here](https://cloud.cerebras.ai/).

Make sure to set the `CEREBRAS_API_KEY` as an alias in your shell.

```bash
export CEREBRAS_API_KEY=<your-api-key>
```

Once you've done so, you can use the following code to get started.

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider("cerebras/gpt-oss-120b")


class Person(BaseModel):
    name: str
    age: int


resp = client.create(
    model="gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Extract the name and age of the person in this sentence: John Smith is 29 years old.",
        }
    ],
    response_model=Person,
)

print(resp)
#> Person(name='John Smith', age=29)
```

We support both the `AsyncCerebras` and `Cerebras` clients.

### Streaming

We also support streaming with the Cerebras client with the `CEREBRAS_JSON` mode so that you can take advantage of Cerebras’s inference speeds and process the response as it comes in.

```python
import instructor
from cerebras.cloud.sdk import Cerebras
from pydantic import BaseModel
from typing import Iterable

client = instructor.from_cerebras(Cerebras(), mode=instructor.Mode.MD_JSON)


class Person(BaseModel):
    name: str
    age: int


resp = client.create(
    model="gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Extract all users from this sentence : Chris is 27 and lives in San Francisco, John is 30 and lives in New York while their college roommate Jessica is 26 and lives in London",
        }
    ],
    response_model=Iterable[Person],
    stream=True,
)

for person in resp:
    print(person)
    #> Person(name='Chris', age=27)
    #> Person(name='John', age=30)
    #> Person(name='Jessica', age=26)
```

And that’s it! We're excited to see what you build with Instructor and Cerebras! If you have any questions about Cerebras or need to get off the API key waitlist, please reach out to sarah.chieng@cerebras.net.
