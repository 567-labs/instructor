---
title: Seamless Fine-Tuning of Python Functions Using Instructor's Distillation
description: Learn how to fine-tune language models with Python functions using Instructor's `Instructions` for efficient data preparation and logging.
---

## See Also

- [Response Models](./models.md) - Working with Pydantic models
- [Validation](./validation.md) - Ensuring output quality
- [Types](./types.md) - Working with different data types
- [Custom Validators](../learning/validation/custom_validators.md) - Build custom validation logic

# Distilling python functions into LLM

`Instructions` from the `Instructor` library offers a seamless way to make language models backward compatible with existing Python functions. By employing Pydantic type hints, it not only ensures compatibility but also facilitates fine-tuning `gpt-4.1-mini` to emulate these functions end-to-end.

If you want to see the full example checkout [examples/distillation](https://github.com/jxnl/instructor/tree/main/examples/distilations)

## The Challenges in Function-Level Fine-Tuning

Replicating the behavior of a Python function in a language model involves intricate data preparation. For instance, teaching a model to execute three-digit multiplication is not as trivial as implementing `def f(a, b): return a * b`. OpenAI's fine-tuning script coupled with their function calling utility provides a structured output, thereby simplifying the data collection process. Additionally, this eliminates the need for passing the schema to the model, thus conserving tokens.

## The Role of `Instructions` in Simplifying the Fine-Tuning Process

By using `Instructions`, you can annotate a Python function that returns a Pydantic object, thereby automating the dataset creation for fine-tuning. A handler for logging is all that's needed to build this dataset.

## How to Implement `Instructions` in Your Code

## Quick Start: How to Use Instructor's Distillation Feature

Before we dig into the nitty-gritty, let's look at how easy it is to use Instructor's distillation feature to use function calling finetuning to export the data to a JSONL file.

```python
import logging
import random
from pydantic import BaseModel

# Logging setup
logging.basicConfig(level=logging.INFO)

from instructor import Instructions, FinetuneFormat  # pip install instructor

instructions = Instructions(
    name="three_digit_multiply",
    finetune_format=FinetuneFormat.MESSAGES,  # or FinetuneFormat.RAW
    # log handler is used to save the data to a file
    # you can imagine saving it to a database or other storage
    # based on your needs!
    log_handlers=[logging.FileHandler("math_finetunes.jsonl")],
)


class Multiply(BaseModel):
    a: int
    b: int
    result: int


# Define a function with distillation
# The decorator will automatically generate a dataset for fine-tuning
# They must return a pydantic model to leverage function calling
@instructions.distil
def fn(a: int, b: int) -> Multiply:
    resp = a * b
    return Multiply(a=a, b=b, result=resp)


# Generate some data
for _ in range(10):
    random.seed(42)
    a = random.randint(100, 999)
    b = random.randint(100, 999)
    print(fn(a, b))
    #> a=754 b=214 result=161356
    #> a=754 b=214 result=161356
    #> a=754 b=214 result=161356
    #> a=754 b=214 result=161356
    #> a=754 b=214 result=161356
    #> a=754 b=214 result=161356
    #> a=754 b=214 result=161356
    #> a=754 b=214 result=161356
    #> a=754 b=214 result=161356
    #> a=754 b=214 result=161356
```

## The Intricacies of Fine-tuning Language Models

Fine-tuning isn't just about writing a function like `def f(a, b): return a * b`. It requires detailed data preparation and logging. However, Instructor provides a built-in logging feature and structured outputs to simplify this.

## Why Instructor and Distillation are Game Changers

The library offers two main benefits:

1. **Efficiency**: Streamlines functions, distilling requirements into model weights and a few lines of code.
2. **Integration**: Eases combining classical machine learning and language models by providing a simple interface that wraps existing functions.

## Role of Instructor in Simplifying Fine-Tuning

The `from instructor import Instructions` feature is a time saver. It auto-generates a fine-tuning dataset, making it a breeze to imitate a function's behavior.

## FinetuneFormat Options

The `finetune_format` parameter controls how the fine-tuning data is structured. There are two options:

### MESSAGES Format (Default)

The `MESSAGES` format creates data in OpenAI's chat completion format with messages and function calls. This is the recommended format for most use cases as it matches OpenAI's fine-tuning API format.

```python
from instructor import Instructions, FinetuneFormat

instructions = Instructions(
    name="my_function",
    finetune_format=FinetuneFormat.MESSAGES,
    log_handlers=[logging.FileHandler("output.jsonl")],
)
```

### RAW Format

The `RAW` format creates a simpler format with function metadata, arguments, and response. Use this format if you need more control over the data structure or are using a custom fine-tuning pipeline.

```python
from instructor import Instructions, FinetuneFormat

instructions = Instructions(
    name="my_function",
    finetune_format=FinetuneFormat.RAW,
    log_handlers=[logging.FileHandler("output.jsonl")],
)
```

## Logging Output and Running a Finetune

Here's how the logging output would look for MESSAGES format:

```python
{
    "messages": [
        {"role": "system", "content": 'Predict the results of this function: ...'},
        {"role": "user", "content": 'Return fn(133, b=539)'},
        {
            "role": "assistant",
            "function_call": {
                "name": "Multiply",
                "arguments": '{"a":133,"b":539,"result":89509}',
            },
        },
    ],
    "functions": [
        {"name": "Multiply", "description": "Correctly extracted `Multiply`..."}
    ],
}
```

For RAW format, the output would look like:

```python
{
    "fn_name": "three_digit_multiply",
    "fn_repr": "def fn(a: int, b: int) -> Multiply:\n    ...",
    "args": [133],
    "kwargs": {"b": 539},
    "response": {"a": 133, "b": 539, "result": 89509},
}
```

Run a finetune like this:

```bash
instructor jobs create-from-file math_finetunes.jsonl
```

Once a model is trained you can simply change `mode` to `dispatch` and it will use the model to run the function!

```python
from instructor import Instructions
from pydantic import BaseModel


class Multiply(BaseModel):
    a: int
    b: int
    result: int


instructions = Instructions(
    name="three_digit_multiply",
)


@instructions.distil(model='gpt-4.1-mini:finetuned-123', mode="dispatch")
def fn(a: int, b: int) -> Multiply:
    # now this code will be short circuited and the model will be used instead.
    resp = a + b
    return Multiply(a=a, b=b, result=resp)
```

With this, you can swap the function implementation, making it backward compatible. You can even imagine using the different models for different tasks or validating and running evals by using the original function and comparing it to the distillation.
