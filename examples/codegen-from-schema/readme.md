# FastAPI Code Generator

## Overview

Generates FastAPI application code from API path, task name, JSON schema path, and Jinja2 prompt template. Also creates a `models.py` file for Pydantic models.

## Dependencies

- FastAPI
- Pydantic
- Jinja2
- datamodel-code-generator

## Functions

### `create_app(api_path: str, task_name: str, json_schema_path: str, prompt_template: str) -> str`

Main function to generate FastAPI application code.

## Usage

Run the script with required parameters.

Example:

```python
fastapi_code = create_app(
    api_path="/api/v1/extract_person",
    task_name="extract_person",
    json_schema_path="./input.json",
    prompt_template="Extract the person from the following: {{biography}}",
)
```

Outputs FastAPI application code to `./run.py` and a Pydantic model to `./models.py`.