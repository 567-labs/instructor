---
authors:
- jxnl
categories:
- LLM Techniques
comments: true
date: 2023-10-17
description: Explore Instructor for fine-tuning language models with Python, simplifying
  function calls, and enhancing performance.
draft: false
tags:
- Instructor
- Fine-tuning
- Python
- Language Models
- Distillation
---

# Enhancing Python Functions with Instructor: A Guide to Fine-Tuning and Distillation

## Introduction

Get ready to dive deep into the world of fine-tuning task specific language models with Python functions. We'll explore how the `instructor.instructions` streamlines this process, making the task you want to distil more efficient and powerful while preserving its original functionality and backwards compatibility.

If you want to see the full example checkout [examples/distillation](https://github.com/jxnl/instructor/tree/main/examples/distilations)

<!-- more -->

## Why use Instructor?

Imagine you're developing a backend service that uses a mix old and new school ML practises, it may involve pipelines with multiple function calls, validations, and data processing. Sounds cumbersome, right? That's where `Instructor` comes in. It simplifies complex procedures, making them more efficient and easier to manage by adding a decorator to your function that will automatically generate a dataset for fine-tuning and help you swap out the function implementation.

## Quick Start: How to Use Instructor's Distillation Feature

Before we dig into the nitty-gritty, let's look at how easy it is to use Instructor's distillation feature to use function calling finetuning to export the data to a JSONL file.

```python
import logging
import random
from pydantic import BaseModel
from instructor import Instructions  # pip install instructor

# Logging setup
logging.basicConfig(level=logging.INFO)

instructions = Instructions(
    name="three_digit_multiply",
    finetune_format="messages",
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
    a = random.randint(100, 999)
    b = random.randint(100, 999)
    print(fn(a, b))
    #> a=268 b=548 result=146864
    #> a=774 b=447 result=345978
    #> a=154 b=902 result=138908
    #> a=304 b=808 result=245632
    #> a=980 b=104 result=101920
    #> a=725 b=455 result=329875
    #> a=206 b=386 result=79516
    #> a=488 b=920 result=448960
    #> a=989 b=889 result=879221
    #> a=815 b=343 result=279545
```

## The Intricacies of Fine-tuning Language Models

Fine-tuning isn't just about writing a function like `def f(a, b): return a * b`. It requires detailed data preparation and logging. However, Instructor provides a built-in logging feature and structured outputs to simplify this.

## Why Instructor and Distillation are Game Changers

The library offers two main benefits:

1. **Efficiency**: Streamlines functions, distilling requirements into model weights and a few lines of code.
2. **Integration**: Eases combining classical machine learning and language models by providing a simple interface that wraps existing functions.

## Role of Instructor in Simplifying Fine-Tuning

The `from instructor import Instructions` feature is a time saver. It auto-generates a fine-tuning dataset, making it a breeze to imitate a function's behavior.

## Logging Output and Running a Finetune

Here's how the logging output would look:

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

Run a finetune like this:

!!! note annotate "Don't forget to set your OpenAI Key as an environment variable"

    All of the `instructor jobs` commands assume you've set an environment variable of `OPENAI_API_KEY` in your shell. You can set this by running the command `export OPENAI_API_KEY=<Insert API Key Here>` in your shell

```bash
instructor jobs create-from-file math_finetunes.jsonl
```

## Next Steps and Future Plans

Here's a sneak peek of what I'm planning:

```python
from instructor import Instructions, patch

patch()  # (1)!


class Multiply(BaseModel):
    a: int
    b: int
    result: int


instructions = Instructions(
    name="three_digit_multiply",
)


@instructions.distil(model='gpt-5.4-mini:finetuned-123', mode="dispatch")  # (2)!
def fn(a: int, b: int) -> Multiply:
    resp = a + b
    return Multiply(a=a, b=b, result=resp)
```

1.  Don't forget to run the `patch()` command that we provide with the `Instructor` package. This helps
    automatically serialize the content back into the `Pydantic`` model that we're looking for.

2.  Don't forget to replace this with your new model id. OpenAI identifies fine tuned models with an id
    of `ft:gpt-5.4-mini:personal::<id>` under their **Fine-tuning** tab on their dashboard

With this, you can swap the function implementation, making it backward compatible. You can even imagine using the different models for different tasks or validating and running evals by using the original function and comparing it to the distillation.

## Conclusion

We've seen how `Instructor` can make your life easier, from fine-tuning to distillation. Now if you're thinking wow, I'd love a backend service to do this for continuously, you're in luck! Please check out the survey at [useinstructor.com](https://useinstructor.com) and let us know who you are.

If you enjoy the content or want to try out `instructor` please check out the [github](https://github.com/jxnl/instructor) and give us a star!