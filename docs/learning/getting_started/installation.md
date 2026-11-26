---
title: Set up provider SDKs and API keys for Instructor
description: Choose the Python package and API key for OpenAI, Claude, Gemini, Cohere, or Mistral. Test your Instructor client with a structured extraction.
---

# Set up your LLM provider {#instructor-installation-guide-setup-for-llm-structured-outputs}

Once you've [installed Instructor](../../installation.md), choose a provider and set its credentials in the environment where you run Python.

## Choose an SDK and API key

The base install includes the OpenAI SDK. For other providers, install the extra listed below. If you use a uv project, replace `python -m pip install` with `uv add`.

| Provider | Install command | Environment variable |
| --- | --- | --- |
| [OpenAI](../../integrations/openai.md) | `python -m pip install instructor` | `OPENAI_API_KEY` |
| [Anthropic Claude](../../integrations/anthropic.md) | `python -m pip install "instructor[anthropic]"` | `ANTHROPIC_API_KEY` |
| [Google Gemini](../../integrations/google.md) | `python -m pip install "instructor[google-genai]"` | `GOOGLE_API_KEY` |
| [Cohere](../../integrations/cohere.md) | `python -m pip install "instructor[cohere]"` | `CO_API_KEY` |
| [Mistral](../../integrations/mistral.md) | `python -m pip install "instructor[mistral]"` | `MISTRAL_API_KEY` |

For [LiteLLM](../../integrations/litellm.md), install `instructor[litellm]` and configure credentials for the underlying provider. See [all integrations](../../integrations/index.md) for local models and cloud platforms.

## Set your API key

For the OpenAI example below, replace the placeholder with your key. Run Python from the same shell.

=== "macOS or Linux"

    ```bash
    export OPENAI_API_KEY="your-openai-key"
    ```

=== "Windows PowerShell"

    ```powershell
    $env:OPENAI_API_KEY = "your-openai-key"
    ```

## Test a request

This example sends a request to OpenAI and may incur API charges. For a local check, [print the installed version](../../installation.md#check-the-installed-version).

```python
import instructor
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-6-luna")
person = client.create(
    response_model=Person,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)

print(person.name, person.age)
```

To use Claude, install the Anthropic extra, set `ANTHROPIC_API_KEY`, and change the provider string to `anthropic/claude-haiku-5-5`.

If you're using a notebook, check that the kernel uses the environment containing the provider SDK. For authentication errors, check that the key is available to the Python process and that your account has access to the model.

Next, [extract your first record](first_extraction.md) or learn how to [define a response model](response_models.md).
