from __future__ import annotations

from openai import OpenAI
import instructor

client = instructor.from_openai(OpenAI())


def fizzbuzz_gpt(n) -> list[int | str]:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=list[int | str],
        messages=[
            {
                "role": "user",
                "content": f"Return the first {n} numbers in fizzbuzz",
            },
        ],
    )  # type: ignore


if __name__ == "__main__":
    print(fizzbuzz_gpt(n=15))
    # > [1, 2, 'Fizz', 4, 'Buzz', 'Fizz', 7, 8, 'Fizz', 'Buzz', 11, 'Fizz', 13, 14, 'FizzBuzz']
