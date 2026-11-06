# AI Agent Instructions: Creating a New Instructor Provider

**Instructions for AI coding agents to create a new provider for the instructor library.**

Copy these instructions to your AI coding agent when you want to add a new LLM provider to instructor. The agent will have everything needed to implement a complete, working provider.

**For human contributors:** See [Adding a New Provider](instructor/v2/README.md#adding-a-new-provider).

---

## Mission

Create a complete, production-ready provider package for the instructor library that:
- Follows the BaseProvider protocol exactly
- Includes comprehensive tests using transcript fixtures  
- Has proper error handling and validation
- Provides excellent documentation
- Integrates seamlessly with the instructor plugin system

## Prerequisites

Before starting, ensure you have:
- Provider name (e.g., "groq", "perplexity", "fireworks")
- Provider's Python SDK package name and version
- API documentation URL
- Sample API key format (for documentation)
- Knowledge of provider's chat completion API structure

## Step-by-Step Implementation

### Step 1: Project Structure Setup

**Note: This creates a new provider integration that follows instructor's existing patterns, not a separate package.**

Create the following structure in the instructor repository:

```
instructor/providers/{provider}/
├── __init__.py              # Empty or basic exports
├── client.py                # from_{provider} function implementation  
└── utils.py                 # Provider-specific utilities

tests/llm/test_{provider}/
├── __init__.py              # Empty
├── conftest.py              # Test configuration & API key handling
├── util.py                  # Models and modes configuration
├── test_simple.py           # Basic functionality tests
├── test_stream.py           # Streaming tests (if supported)
├── test_format.py           # Format/structure tests
└── test_retries.py          # Error handling tests

docs/integrations/
└── {provider}.md            # Provider documentation following existing pattern
```

**Important: You're adding to the existing instructor codebase, not creating a separate package.**

### Step 2: Provider Client Implementation

#### File: `instructor/providers/{provider}/client.py`

Follow the exact pattern used by other providers in instructor. This creates a `from_{provider}` function:

```python
from __future__ import annotations

from typing import Any, overload

import instructor
from ...core.client import AsyncInstructor, Instructor

# Import the provider's SDK
from {provider_sdk} import {SyncClient}, {AsyncClient}  # Replace with actual imports


@overload
def from_{provider}(
    client: {SyncClient},
    mode: instructor.Mode = instructor.Mode.{PROVIDER}_TOOLS,  # Default mode
    **kwargs: Any,
) -> Instructor: ...


@overload  
def from_{provider}(
    client: {AsyncClient},
    mode: instructor.Mode = instructor.Mode.{PROVIDER}_TOOLS,  # Default mode
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_{provider}(
    client: {SyncClient} | {AsyncClient},
    mode: instructor.Mode = instructor.Mode.{PROVIDER}_TOOLS,  # Default mode
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """
    Create an instructor client from a {Provider} client
    
    Args:
        client: {Provider} sync or async client instance
        mode: Mode to use for structured outputs
        **kwargs: Additional arguments passed to instructor client
        
    Returns:
        Instructor or AsyncInstructor instance
    """
    # Define valid modes for this provider
    valid_modes = {
        instructor.Mode.{PROVIDER}_TOOLS,
        instructor.Mode.{PROVIDER}_JSON,
        # Add other modes your provider supports
    }

    # Validate mode
    if mode not in valid_modes:
        from ...core.exceptions import ModeError
        raise ModeError(
            mode=str(mode),
            provider="{Provider}",
            valid_modes=[str(m) for m in valid_modes],
        )

    # Validate client type  
    if not isinstance(client, ({AsyncClient}, {SyncClient})):
        from ...core.exceptions import ClientError
        raise ClientError(
            f"Client must be an instance of {SyncClient} or {AsyncClient}. "
            f"Got: {type(client).__name__}"
        )

    # Handle async client
    if isinstance(client, {AsyncClient}):
        
        async def async_wrapper(*args: Any, **kwargs: Any):
            """Wrapper for async client calls"""
            if "stream" in kwargs and kwargs["stream"] is True:
                # Handle streaming if supported
                return client.chat.completions.acreate(*args, **kwargs)
            return await client.chat.completions.acreate(*args, **kwargs)

        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=async_wrapper, mode=mode),
            provider=instructor.Provider.{PROVIDER},  # Must be defined in Provider enum
            mode=mode,
            **kwargs,
        )

    # Handle sync client
    if isinstance(client, {SyncClient}):
        return Instructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mode),
            provider=instructor.Provider.{PROVIDER},  # Must be defined in Provider enum  
            mode=mode,
            **kwargs,
        )
```

### Step 3: Mode Handlers Implementation

#### File: `instructor_{provider}/handlers.py`

```python
"""
Mode handlers for {Provider} provider

Each handler knows how to:
1. Format requests for the specific mode (TOOLS, JSON, etc.)
2. Parse responses back into Pydantic models
3. Handle provider-specific response formats
"""

from typing import Dict, Any, Type, Union
from pydantic import BaseModel
from instructor.mode import Mode
from instructor.function_calls import openai_schema
import json

class BaseModeHandler:
    """Base class for mode handlers"""
    
    def __init__(self, provider):
        self.provider = provider
    
    def prepare_request(
        self, 
        response_model: Type[BaseModel], 
        messages: list, 
        model: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare request for this mode"""
        raise NotImplementedError
    
    def parse_response(self, response: Any, response_model: Type[BaseModel]) -> BaseModel:
        """Parse provider response into Pydantic model"""
        raise NotImplementedError

class ToolsHandler(BaseModeHandler):
    """Handler for function/tool calling mode"""
    
    def prepare_request(self, response_model, messages, model, **kwargs):
        # Convert Pydantic model to function schema
        schema = openai_schema(response_model)
        
        return {
            "model": model,
            "messages": messages,
            "tools": [{
                "type": "function",
                "function": schema
            }],
            "tool_choice": "auto",  # or provider-specific equivalent
            **kwargs
        }
    
    def parse_response(self, response, response_model):
        # Extract function call from response
        # This is provider-specific - adapt to your provider's response format
        
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                return response_model(**function_args)
        
        raise ValueError("No valid tool call found in response")

class JSONHandler(BaseModeHandler):
    """Handler for JSON mode responses"""
    
    def prepare_request(self, response_model, messages, model, **kwargs):
        # Add JSON schema to system message
        schema_prompt = f"""
You must respond with valid JSON that matches this schema:
{response_model.model_json_schema()}

Respond with only the JSON, no additional text.
"""
        
        # Add schema to messages
        enhanced_messages = [
            {"role": "system", "content": schema_prompt}
        ] + messages
        
        return {
            "model": model,
            "messages": enhanced_messages,
            "response_format": {"type": "json_object"},  # if provider supports
            **kwargs
        }
    
    def parse_response(self, response, response_model):
        # Extract JSON from response content
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            try:
                data = json.loads(content)
                return response_model(**data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in response: {e}")
        
        raise ValueError("No valid response content found")

# Handler registry
_HANDLERS = {
    Mode.TOOLS: ToolsHandler,
    Mode.JSON: JSONHandler,
    # Add other modes as supported by provider
}

def get_handler(mode: Mode, provider) -> BaseModeHandler:
    """Get handler instance for the specified mode"""
    if mode not in _HANDLERS:
        supported = ", ".join(h.name for h in _HANDLERS.keys())
        raise ValueError(f"Mode {mode} not supported. Supported modes: {supported}")
    
    handler_class = _HANDLERS[mode]
    return handler_class(provider)
```

### Step 4: Package Configuration

#### File: `pyproject.toml`

```toml
[project]
name = "instructor-{provider}"
version = "0.1.0"
description = "Instructor provider for {Provider Name}"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
license = {text = "MIT"}
requires-python = ">=3.9"
dependencies = [
    "instructor-core>=2.0.0,<3.0.0",
    "{provider_sdk}>=X.X.X,<Y.0.0",  # Replace with actual version constraints
    "pydantic>=2.8.0,<3.0.0",
]

readme = "README.md"
keywords = ["instructor", "llm", "structured-output", "{provider}"]

[project.urls]
Homepage = "https://github.com/instructor-ai/instructor"
Documentation = "https://python.useinstructor.com"
Repository = "https://github.com/instructor-ai/instructor"

[project.optional-dependencies]
dev = [
    "pytest>=8.3.3,<9.0.0",
    "pytest-asyncio>=0.24.0,<1.0.0", 
    "pytest-mock>=3.12.0",
    "responses>=0.24.0",  # For HTTP mocking
    "python-dotenv>=1.0.1",
]

# Register the provider with instructor's plugin system
[project.entry-points."instructor.providers"]
{provider} = "instructor_{provider}:{Provider}Provider"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: Unit tests (fast, no external dependencies)",
    "integration: Integration tests (may require API keys)", 
    "live: Live API tests (requires valid API key)"
]

[tool.ruff]
target-version = "py39"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "B", "A", "C4", "T20"]
ignore = ["E501"]  # Line too long (handled by formatter)
```

### Step 3: Testing Implementation

#### File: `tests/llm/test_{provider}/conftest.py`

Follow the exact pattern used by all other providers:

```python
import os
import pytest

# Skip entire test suite if API key is missing
if not os.getenv("{PROVIDER}_API_KEY"):
    pytest.skip(
        "{PROVIDER}_API_KEY environment variable not set",
        allow_module_level=True,
    )

# Skip if provider package is not installed  
try:
    from {provider_sdk} import {SyncClient}, {AsyncClient}  # Replace with actual imports
except ImportError:
    pytest.skip("{provider_sdk} package is not installed", allow_module_level=True)


@pytest.fixture(scope="function")
def client():
    """Sync client fixture"""
    yield {SyncClient}()


@pytest.fixture(scope="function") 
def aclient():
    """Async client fixture"""
    yield {AsyncClient}()
```

#### File: `tests/llm/test_{provider}/util.py`

Define supported models and modes:

```python
import instructor

# Replace with actual model names your provider supports
models = ["provider-model-name-1", "provider-model-name-2"]

# Replace with actual modes your provider supports
modes = [
    instructor.Mode.{PROVIDER}_TOOLS,
    instructor.Mode.{PROVIDER}_JSON,
]
```

#### File: `tests/llm/test_{provider}/test_simple.py`

Follow the standard pattern for basic functionality tests:

```python
import instructor
from {provider_sdk} import {SyncClient}, {AsyncClient}  # Replace with actual imports
from pydantic import BaseModel, field_validator
import pytest
from itertools import product
from .util import models, modes


class User(BaseModel):
    """Standard test model"""
    name: str
    age: int


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_{provider}_sync(model: str, mode: instructor.Mode, client):
    """Test basic sync functionality"""
    client = instructor.from_{provider}(client, mode=mode)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence: Ivan is 27 and lives in Singapore",
            },
        ],
        response_model=User,
    )

    assert resp.name.lower() == "ivan"
    assert resp.age == 27


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_{provider}_sync_validated(model: str, mode: instructor.Mode, client):
    """Test sync with validation retries"""
    class ValidatedUser(BaseModel):
        name: str
        age: int

        @field_validator("name")
        def name_validator(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"All letters in the name must be uppercase (Eg. JOHN, SMITH) - {v} is not a valid example."
                )
            return v

    client = instructor.from_{provider}(client, mode=mode)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user", 
                "content": "Extract a user from this sentence: Ivan is 27 and lives in Singapore",
            },
        ],
        max_retries=5,
        response_model=ValidatedUser,
    )

    assert resp.name == "IVAN"
    assert resp.age == 27


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio(scope="session")
async def test_{provider}_async(model: str, mode: instructor.Mode, aclient):
    """Test async functionality"""
    client = instructor.from_{provider}(aclient, mode=mode)

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence: Ivan is 27 and lives in Singapore",
            },
        ],
        response_model=User,
    )

    assert resp.name.lower() == "ivan"
    assert resp.age == 27


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio(scope="session")
async def test_{provider}_async_validated(model: str, mode: instructor.Mode, aclient):
    """Test async with validation retries"""
    class ValidatedUser(BaseModel):
        name: str
        age: int

        @field_validator("name")
        def name_validator(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"Make sure to uppercase all letters in the name field. Examples include: JOHN, SMITH, etc. {v} is not a valid example."
                )
            return v

    client = instructor.from_{provider}(aclient, mode=mode)

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence: Ivan is 27 and lives in Singapore",
            },
        ],
        response_model=ValidatedUser,
        max_retries=5,
    )

    assert resp.name == "IVAN"
    assert resp.age == 27
```

### Step 4: Required Infrastructure Updates

#### A. Add Mode Constants

Add your provider's modes to `instructor/mode.py`:

```python
# Add to the Mode enum class
{PROVIDER}_TOOLS = "{provider}_tools"
{PROVIDER}_JSON = "{provider}_json"
# Add other modes as needed
```

#### B. Add Provider to Enum

Add your provider to `instructor/utils/providers.py`:

```python
# Add to the Provider enum
{PROVIDER} = "{provider}"
```

#### C. Update Main __init__.py

Add conditional import to `instructor/__init__.py`:

```python
# Add this block with the other provider imports
if importlib.util.find_spec("{provider_sdk}") is not None:
    from .providers.{provider}.client import from_{provider}
    
    __all__ += ["from_{provider}"]
```

#### D. Add to pyproject.toml

Add your provider to the optional dependencies:

```toml
# In [project.optional-dependencies]
{provider} = ["{provider_sdk}>=X.X.X,<Y.0.0"]  # Replace with actual version

# In [dependency-groups] 
{provider} = ["{provider_sdk}>=X.X.X,<Y.0.0"]
```

### Step 5: Documentation

#### File: `docs/integrations/{provider}.md`

Follow the exact pattern of existing provider docs:

```markdown
---
title: "Structured outputs with {Provider}, a complete guide w/ instructor"
description: "Complete guide to using Instructor with {Provider} models. Learn how to generate structured, type-safe outputs with {provider description}."
---

# Structured outputs with {Provider}, a complete guide w/ instructor

{Provider description and benefits}. This guide shows you how to use Instructor with {Provider}'s models for type-safe, validated responses.

## Quick Start

Install Instructor with {Provider} support:

```bash
pip install "instructor[{provider}]"
```

## Simple User Example (Sync)

```python
from {provider_sdk} import {SyncClient}
import instructor
from pydantic import BaseModel

# Initialize the client
client = {SyncClient}()

# Enable instructor patches
client = instructor.from_{provider}(client)

class User(BaseModel):
    name: str
    age: int

# Extract structured data
user = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    response_model=User
)

print(user.name)  # Jason
print(user.age)   # 25
```

## Simple User Example (Async)

```python
from {provider_sdk} import {AsyncClient}
import instructor
from pydantic import BaseModel
import asyncio

# Initialize async client
client = {AsyncClient}()

# Enable instructor patches
client = instructor.from_{provider}(client)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.chat.completions.create(
        model="your-model-name",
        messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
        response_model=User
    )
    return user

# Run async function
user = asyncio.run(extract_user())
print(user.name)  # Jason
print(user.age)   # 25
```

## Supported Models

- `model-1` - Description and capabilities
- `model-2` - Description and capabilities

Check [{Provider} documentation](provider-docs-url) for the complete list of available models.

## Modes

The {Provider} provider supports these modes:

- `instructor.Mode.{PROVIDER}_TOOLS` - Uses {provider} function calling (recommended)
- `instructor.Mode.{PROVIDER}_JSON` - Uses JSON mode responses

```python
client = instructor.from_{provider}(client, mode=instructor.Mode.{PROVIDER}_TOOLS)
```

## Advanced Usage

### Validation and Retries

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str
    age: int
    
    @field_validator('age')
    def validate_age(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v

# Automatic retries on validation errors
user = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Extract: Jason is -5 years old"}],
    response_model=User,
    max_retries=3
)
```

### Complex Nested Models

```python
from typing import List

class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    name: str
    age: int
    addresses: List[Address]

users = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Extract user info with multiple addresses..."}],
    response_model=User
)
```

## Migration from Other Providers

If you're migrating from another provider:

```python
# Old way (other provider)
# client = instructor.from_openai(openai_client)

# New way ({Provider})  
client = instructor.from_{provider}({provider_sdk}.{SyncClient}())
```

## API Reference

For detailed API documentation, see the [Instructor API reference](../api.md).
```

## Example Provider: Groq

Here's a concrete example implementing a Groq provider:

#### File: `instructor/providers/groq/client.py`
```python
from __future__ import annotations
from typing import Any, overload
import instructor
from ...core.client import AsyncInstructor, Instructor
from groq import Groq, AsyncGroq

@overload
def from_groq(
    client: Groq,
    mode: instructor.Mode = instructor.Mode.GROQ_TOOLS,
    **kwargs: Any,
) -> Instructor: ...

@overload  
def from_groq(
    client: AsyncGroq,
    mode: instructor.Mode = instructor.Mode.GROQ_TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...

def from_groq(
    client: Groq | AsyncGroq,
    mode: instructor.Mode = instructor.Mode.GROQ_TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    valid_modes = {
        instructor.Mode.GROQ_TOOLS,
        instructor.Mode.GROQ_JSON,
    }

    if mode not in valid_modes:
        from ...core.exceptions import ModeError
        raise ModeError(
            mode=str(mode),
            provider="Groq",
            valid_modes=[str(m) for m in valid_modes],
        )

    if not isinstance(client, (AsyncGroq, Groq)):
        from ...core.exceptions import ClientError
        raise ClientError(
            f"Client must be an instance of Groq or AsyncGroq. "
            f"Got: {type(client).__name__}"
        )

    if isinstance(client, AsyncGroq):
        async def async_wrapper(*args: Any, **kwargs: Any):
            return await client.chat.completions.acreate(*args, **kwargs)

        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=async_wrapper, mode=mode),
            provider=instructor.Provider.GROQ,
            mode=mode,
            **kwargs,
        )

    return Instructor(
        client=client,
        create=instructor.patch(create=client.chat.completions.create, mode=mode),
        provider=instructor.Provider.GROQ,
        mode=mode,
        **kwargs,
    )
```

## Quality Checklist

Before submitting your provider implementation, verify:

### Core Implementation
- [ ] `from_{provider}` function implemented following the exact pattern
- [ ] Both sync and async clients supported with proper overloads
- [ ] Valid modes defined and enforced with proper error messages
- [ ] Client type validation with helpful error messages
- [ ] Proper use of `instructor.patch()` for both sync and async

### Testing
- [ ] `conftest.py` skips tests if API key missing or package not installed
- [ ] `util.py` defines supported models and modes
- [ ] `test_simple.py` covers basic sync/async functionality with validation
- [ ] Tests use parametrized approach with `product(models, modes)`
- [ ] All tests pass with real API key: `pytest tests/llm/test_{provider}/`

### Infrastructure Updates
- [ ] Modes added to `instructor/mode.py`
- [ ] Provider added to `instructor/utils/providers.py` Provider enum
- [ ] Conditional import added to `instructor/__init__.py`
- [ ] Dependencies added to `pyproject.toml` optional-dependencies
- [ ] Dependencies added to `pyproject.toml` dependency-groups

### Documentation
- [ ] Provider documentation created in `docs/integrations/{provider}.md`
- [ ] Follows exact pattern with frontmatter, examples, and sections
- [ ] All code examples are tested and work
- [ ] Covers sync/async usage, validation, nested models
- [ ] Links to provider documentation and API reference

### Integration
- [ ] Works with existing instructor patterns and conventions
- [ ] Error messages are helpful and actionable
- [ ] Follows the same API as other providers
- [ ] No performance regressions

## Submission Process

1. **Test Locally**: Ensure all tests pass and examples work
2. **Create PR**: Submit to instructor repository
3. **Package Registry**: Publish to PyPI as `instructor-{provider}`
4. **Documentation**: Add to instructor docs site
5. **Announcement**: Share with community

## Common Issues & Solutions

### "Provider not found" error
- Check entry point configuration in pyproject.toml
- Verify provider name matches exactly
- Ensure package is installed in same environment

### Validation errors not retrying
- Verify error handling in chat() method catches ValidationError
- Check that validation messages are added to conversation
- Ensure max_retries parameter is respected

### Mode not supported
- Implement handler in handlers.py for the mode
- Add to _HANDLERS registry
- Test with provider's actual API capabilities

### Streaming issues
- Check if provider supports streaming at all
- Implement incremental parsing for partial responses
- Handle stream interruption and reconnection

### Type checking failures  
- Ensure all method signatures match BaseProvider protocol exactly
- Add proper type hints for all parameters and returns
- Use Union/Optional types where appropriate

---

**This completes the full provider implementation guide. Follow these instructions systematically and you'll have a production-ready instructor provider that integrates seamlessly with the existing ecosystem.**
