---
title: Install Instructor
description: Check Python compatibility, install Instructor with pip or uv, and verify the version. The base package includes Pydantic and the OpenAI SDK.
---

# Install Instructor for Python

Instructor 1.18 requires Python 3.10 or later. If you're upgrading from 1.17 on Python 3.9, create an environment with a newer Python interpreter first.

## Install the package

=== "pip"

    In an activated virtual environment, run

    ```bash
    python -m pip install instructor
    ```

=== "uv"

    In a uv project, run

    ```bash
    uv add instructor
    ```

The base package includes Pydantic and the OpenAI SDK. For Claude, Gemini, or another provider, see [provider SDKs and API keys](./learning/getting_started/installation.md).

## Check the installed version

These commands import Instructor locally. You don't need an API key.

=== "pip"

    ```bash
    python -c "import instructor; print(instructor.__version__)"
    ```

=== "uv"

    ```bash
    uv run python -c "import instructor; print(instructor.__version__)"
    ```

If Python can't find Instructor, check that your editor or notebook uses the same environment where you installed it.

Next, [extract and validate your first record](./getting-started.md).
