# Instructor Hooks Example

This example demonstrates how to use the Hooks system in the Instructor library to monitor, log, and debug your LLM interactions.

## What are Hooks?

Hooks provide a powerful mechanism for intercepting and handling events during the completion and parsing process. They allow you to add custom behavior, logging, or error handling at various stages of the API interaction.

The Instructor library supports several predefined hooks:

- `completion:kwargs`: Emitted when completion arguments are provided
- `completion:response`: Emitted when a completion response is received
- `completion:error`: Emitted when an error occurs during completion
- `completion:last_attempt`: Emitted when the last retry attempt is made
- `parse:error`: Emitted when an error occurs during response parsing

## What This Example Shows

This example demonstrates:

1. **Basic Hook Registration**: How to register handlers for different hook events
2. **Multiple Handlers**: How to register multiple handlers for the same event
3. **Statistics Collection**: How to collect and track API usage statistics
4. **Error Handling**: How to catch and process different types of errors
5. **Hook Cleanup**: How to remove hooks when they're no longer needed

## Usage Examples

The code demonstrates three scenarios:

1. **Successful Extraction**: A basic example that works correctly
2. **Parse Error**: An example that triggers a validation error
3. **Multiple Hooks**: Shows how to attach multiple handlers to the same event

## How to Run the Example

```bash
# Navigate to the hooks example directory
cd examples/hooks

# Run the example
python run.py
```

## Expected Output

The example will print detailed information about each request, including:

- 🔍 Request details (model, prompt)
- 📏 Approximate input token count
- 📊 Token usage statistics
- ✅ Successful responses
- ⚠️ Parse errors
- ❌ Completion errors
- 🔄 Retry attempt notifications

At the end, it will print a summary of the statistics collected.

## Learn More

For more information about hooks in Instructor, see the [hooks documentation](https://python.useinstructor.com/concepts/hooks/). 