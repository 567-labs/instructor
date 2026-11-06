import logging

from pydantic import BaseModel, Field
from instructor import Instructions

logging.basicConfig(level=logging.INFO)

# Usage
instructions = Instructions(
    name="three_digit_multiply",
    finetune_format="messages",
    log_handlers=[
        logging.FileHandler("math_finetunes.jsonl"),
    ],
)


class Multiply(BaseModel):
    a: int
    b: int
    result: int = Field(..., description="The result of the multiplication")


@instructions.distil
def fn(a: int, b: int) -> Multiply:
    """Return the result of multiplying a and b together"""
    resp = a * b
    return Multiply(a=a, b=b, result=resp)


if __name__ == "__main__":
    import random

    log_lines = {
        "messages": [
            {
                "role": "system",
                "content": 'Predict the results of this function:\n\ndef fn(a: int, b: int) -> __main__.Multiply\n"""\nReturn the result of multiplying a and b together\n"""',
            },
            {"role": "user", "content": "Return `fn(169, b=166)`"},
            {
                "role": "assistant",
                "function_call": {
                    "name": "Multiply",
                    "arguments": '{\n  "a": 169,\n  "b": 166,\n  "result": 28054\n}',
                },
            },
        ],
        "functions": [
            {
                "name": "Multiply",
                "description": "Correctly extracted `Multiply` with all the required parameters with correct types",
                "parameters": {
                    "properties": {
                        "a": {"title": "A", "type": "integer"},
                        "b": {"title": "B", "type": "integer"},
                        "result": {
                            "description": "The result of the multiplication",
                            "title": "Result",
                            "type": "integer",
                        },
                    },
                    "required": ["a", "b", "result"],
                    "type": "object",
                },
            }
        ],
    }
    for _ in range(10):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        print("returning", fn(a, b=b))
