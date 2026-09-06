---
description: Discover why Instructor is the simplest, most reliable way to get structured outputs from LLMs.
---

# Why use Instructor?

You've built something with an LLM, but 15% of the time it returns garbage. Parsing JSON is a nightmare. Different providers have different APIs. There has to be a better way.

## The pain of unstructured outputs

Let's be honest about what working with LLMs is really like:

```python
# What you want:
user_info = extract_user("John is 25 years old")
print(user_info.name)  # "John"
print(user_info.age)   # 25

# What you actually get:
response = llm.complete("Extract: John is 25 years old")
# "I'd be happy to help! Based on the text, the user's name is John
# and their age is 25. Is there anything else you'd like me to extract?"

# Now you need to:
# 1. Parse this text somehow
# 2. Handle when it returns JSON with syntax errors
# 3. Validate the data matches what you expect
# 4. Retry when it fails (which it will)
# 5. Do this differently for each LLM provider
```

## The Instructor difference

Here's the same task with Instructor:

```python
import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

client = instructor.from_provider("openai/gpt-5.4-mini")
user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25 years old"}],
)

print(user.name)  # "John"
print(user.age)   # 25
```

**That's it.** No parsing. No retries. No provider-specific code.

## Real problems Instructor solves

### 1. "It works 90% of the time"

Without Instructor, your LLM returns perfect JSON most of the time. But that 10% will ruin your weekend.

```python
# Without Instructor: Brittle code that breaks randomly
try:
    data = json.loads(llm_response)
    user = User(**data)  # KeyError: 'name'
except:
    # Now what? Retry? Parse the text? Give up?
    pass

# With Instructor: Automatic retries with validation errors
user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
    max_retries=3,  # Retries with validation errors
)
# Always returns valid User object or raises clear exception
```

### 2. "Each provider is different"

Every LLM provider has its own API. Your code becomes a mess of conditionals.

```python
# Without Instructor: Provider-specific spaghetti
if provider == "openai":
    response = openai.chat.completions.create(
        tools=[{"type": "function", "function": {...}}]
    )
    data = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
elif provider == "anthropic":
    response = anthropic.messages.create(
        tools=[{"name": "extract", "input_schema": {...}}]
    )
    data = response.content[0].input
elif provider == "google":
    # ... different API again

# With Instructor: One API for all providers
client = instructor.from_provider("openai/gpt-5.4-mini")
# or
client = instructor.from_provider("anthropic/claude-3")
# or
client = instructor.from_provider("google/gemini-pro")

# Same code for all providers
user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

### 3. "Complex data structures are impossible"

Nested objects, lists, enums - LLMs struggle with complex schemas.

```python
# Without Instructor: Good luck with this
schema = {
    "type": "object",
    "properties": {
        "users": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "addresses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }
}

# With Instructor: Just use Python
from typing import List

class Address(BaseModel):
    street: str
    city: str

class User(BaseModel):
    name: str
    addresses: List[Address]

class UserList(BaseModel):
    users: List[User]

# Works perfectly
result = client.create(
    response_model=UserList,
    messages=[{"role": "user", "content": "..."}],
)
```

## The cost of not using Instructor

Let's talk real numbers:

**Time wasted:**
- 2-3 hours implementing JSON parsing and validation
- 4-6 hours debugging edge cases
- 2-3 hours for each new provider you add
- Ongoing maintenance as APIs change

**Bugs in production:**
- Malformed JSON crashes your app
- Missing fields cause silent failures
- Type mismatches corrupt your database
- Customer complaints about reliability

**Developer frustration:**
- "It worked in testing!"
- "Why is the JSON different this time?"
- "How do I handle when it returns a string instead of a number?"

## What developers are saying

Based on our GitHub issues and Discord:

- **"Reduced our LLM code by 80%"** - Common feedback
- **"Finally, LLM outputs I can trust"** - From production users
- **"The retries alone are worth it"** - Saves hours of edge-case handling
- **"Works exactly the same with every provider"** - No more provider lock-in

## Start now, thank yourself later

Every day without Instructor is another day of:
- Debugging malformed JSON
- Writing provider-specific code
- Handling validation manually
- Explaining to your PM why the LLM integration is flaky

Install Instructor:
```bash
pip install instructor
```

Try it in 30 seconds:
```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-5.4-mini")

class User(BaseModel):
    name: str
    age: int

user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25 years old"}],
)

print(user)  # User(name='John', age=25)
```

## When NOT to use Instructor

Let's be clear - you might not need Instructor if:

- You only need raw text responses (chatbots, creative writing)
- You're building a one-off script with no error handling
- You enjoy debugging JSON parsing errors at 3am

For everyone else building production LLM applications, Instructor is the obvious choice.

[Get Started →](index.md#quick-start){ .md-button .md-button--primary }