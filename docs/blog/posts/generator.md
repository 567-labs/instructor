---
authors:
- jxnl
- anmol
categories:
- LLM Techniques
comments: true
date: 2023-11-26
description: Explore Python generators and their role in enhancing LLM streaming for
  improved latency and user experience in applications.
draft: false
slug: python-generators-and-llm-streaming
tags:
- Python
- Generators
- LLM Streaming
- Data Processing
- Performance Optimization
---

# Generators and LLM Streaming

Latency is crucial, especially in eCommerce and newer chat applications like ChatGPT. Streaming is the solution that enables us to enhance the user experience without the need for faster response times.

And what makes streaming possible? Generators!

<!-- more -->

In this post, we're going to dive into the cool world of Python generators - these tools are more than just a coding syntax trick. We'll explore Python generators from the ground up and then delve into LLM streaming using the Instructor library.

## Python Generators: An Efficient Approach to Iterables

Generators in Python are a game-changer for handling large data sets and stream processing. They allow functions to yield values one at a time, pausing and resuming their state, which is a faster and more memory-efficient approach compared to traditional collections that store all elements in memory.

### The Basics: Yielding Values

A generator function in Python uses the `yield` keyword. It yields values one at a time, allowing the function to pause and resume its state.

```python
def count_to_3():
    yield 1
    yield 2
    yield 3


for num in count_to_3():
    print(num)
    #> 1
    #> 2
    #> 3
```

```
1
2
3
```

### Advantages Over Traditional Collections

- **Lazy Evaluation & reduced latency**: The time to get the first element (or time-to-first-token in LLM land) from a generator is significantly lower. Generators only produce one value at a time, whereas accessing the first element of a collection will require that the whole collection be created first.
- **Memory Efficiency**: Only one item is in memory at a time.
- **Maintain State**: Automatically maintains state between executions.

Let's see how much faster generators are and where they really shine:

```python
import time


def expensive_func(x):
    """Simulate an expensive operation."""
    time.sleep(1)
    return x**2


def calculate_time_for_first_result_with_list(func_input, func):
    """Calculate using a list comprehension and return the first result with its computation time."""
    start_perf = time.perf_counter()
    result = [func(x) for x in func_input][0]
    end_perf = time.perf_counter()
    print(f"Time for first result (list): {end_perf - start_perf:.2f} seconds")
    #> Time for first result (list): 5.02 seconds
    return result


def calculate_time_for_first_result_with_generator(func_input, func):
    """Calculate using a generator and return the first result with its computation time."""
    start_perf = time.perf_counter()
    result = next(func(x) for x in func_input)
    end_perf = time.perf_counter()
    print(f"Time for first result (generator): {end_perf - start_perf:.2f} seconds")
    #> Time for first result (generator): 1.01 seconds
    return result


# Prepare inputs for the function
numbers = [1, 2, 3, 4, 5]

# Benchmarking
first_result_list = calculate_time_for_first_result_with_list(numbers, expensive_func)
first_result_gen = calculate_time_for_first_result_with_generator(
    numbers, expensive_func
)
```

```
Time for first result (list): 5.02 seconds
Time for first result (generator): 1.01 seconds
```

The generator computes one expensive operation and returns the first result immediately, while the list comprehension computes the expensive operation for all elements in the list before returning the first result.

### Generator Expressions: A Shortcut

Python also allows creating generators in a single line of code, known as generator expressions. They are syntactically similar to list comprehensions but use parentheses.

```python
squares = (x * x for x in range(10))
```

### Use Cases in Real-World Applications

Generators shine in scenarios like reading large files, data streaming (eg. llm token streaming), and pipeline creation for data processing.

## LLM Streaming

If you've used ChatGPT, you'll see that the tokens are streamed out one by one, instead of the full response being shown at the end (can you imagine waiting for the full response??). This is made possible by generators.

Here's how a vanilla openai generator looks:

```python
from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(
    api_key="My API Key",
)

response_generator = client.create(
    model='gpt-5.4-mini',
    messages=[{'role': 'user', 'content': "What are some good reasons to smile?"}],
    temperature=0,
    stream=True,
)

for chunk in response_generator:
    print(chunk.choices[0].delta.content, end="")
```

This is great, but what if we want to do some structured extraction on this stream? For instance, we might want to render frontend components based on product rankings that are streamed out by an LLM.

Should we wait for the entire stream to finish before extracting & validating the list of components or can we extract & validate the components in real time as they are streamed?

In e-commerce, every millisecond matters so the time-to-first-render can differentiate a successful and not-so-successful e commerce store (and i know how a failing e commerce store feels :/ ).

Let's see how we can use Instructor to handle extraction from this real time stream!

### E-commerce Product Ranking

#### Scenario

Imagine an e-commerce platform where we have:

• **a customer profile**: this includes a detailed history of purchases, browsing behavior, product ratings, preferences in various categories, search history, and even responses to previous recommendations. This extensive data is crucial for generating highly personalized and relevant product suggestions.

• **a list of candidate products**: these could be some shortlisted products we think the customer would like.

Our goal is to re-rerank these candidate products for the best conversion and we'll use an LLM!

#### Stream Processing

**User Data**:

Let's assume we have the following user profile:

```python
profile_data = """
Customer ID: 12345
Recent Purchases: [Laptop, Wireless Headphones, Smart Watch]
Frequently Browsed Categories: [Electronics, Books, Fitness Equipment]
Product Ratings: {Laptop: 5 stars, Wireless Headphones: 4 stars}
Recent Search History: [best budget laptops 2023, latest sci-fi books, yoga mats]
Preferred Brands: [Apple, AllBirds, Bench]
Responses to Previous Recommendations: {Philips: Not Interested, Adidas: Not Interested}
Loyalty Program Status: Gold Member
Average Monthly Spend: $500
Preferred Shopping Times: Weekend Evenings
...
"""
```

We want to rank the following products for this user:

```python
products = [
    {
        "product_id": 1,
        "product_name": "Apple MacBook Air (2023) - Latest model, high performance, portable",
    },
    {
        "product_id": 2,
        "product_name": "Sony WH-1000XM4 Wireless Headphones - Noise-canceling, long battery life",
    },
    {
        "product_id": 3,
        "product_name": "Apple Watch Series 7 - Advanced fitness tracking, seamless integration with Apple ecosystem",
    },
    {
        "product_id": 4,
        "product_name": "Kindle Oasis - Premium e-reader with adjustable warm light",
    },
    {
        "product_id": 5,
        "product_name": "AllBirds Wool Runners - Comfortable, eco-friendly sneakers",
    },
    {
        "product_id": 6,
        "product_name": "Manduka PRO Yoga Mat - High-quality, durable, eco-friendly",
    },
    {
        "product_id": 7,
        "product_name": "Bench Hooded Jacket - Stylish, durable, suitable for outdoor activities",
    },
    {
        "product_id": 8,
        "product_name": "GoPro HERO9 Black - 5K video, waterproof, for action photography",
    },
    {
        "product_id": 9,
        "product_name": "Nespresso Vertuo Next Coffee Machine - Quality coffee, easy to use, compact design",
    },
    {
        "product_id": 10,
        "product_name": "Project Hail Mary by Andy Weir - Latest sci-fi book from a renowned author",
    },
]
```

Let's now define our models for structured extraction. Note: instructor will conveniently let us use `Iterable` to model an iterable of our class. In this case, once we define our product recommendation model, we can slap on `Iterable` to define what we ultimately want - a (ranked) list of product recommendations.

```python
import instructor
from openai import OpenAI
from typing import Iterable
from pydantic import BaseModel

client = instructor.from_openai(OpenAI(), mode=instructor.function_calls.Mode.JSON)


class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str


Recommendations = Iterable[ProductRecommendation]
```

Now let's use our instructor patch. Since we don't want to wait for all the tokens to finish, will set stream to `True` and process each product recommendation as it comes in:

```python
prompt = (
    f"Based on the following user profile:\n{profile_data}\nRank the following products from most relevant to least relevant:\n"
    + '\n'.join(
        f"{product['product_id']} {product['product_name']}" for product in products
    )
)

start_perf = time.perf_counter()
recommendations_stream = client.create(
    model="gpt-5.4-mini",
    temperature=0.1,
    response_model=Iterable[ProductRecommendation],
    stream=True,
    messages=[
        {
            "role": "system",
            "content": "Generate product recommendations based on the customer profile. Return in order of highest recommended first.",
        },
        {"role": "user", "content": prompt},
    ],
)
for product in recommendations_stream:
    print(product)
    end_perf = time.perf_counter()
    print(f"Time for first result (generator): {end_perf - start_perf:.2f} seconds")
    break
```

```
product_id='1' product_name='Apple MacBook Air (2023)'
Time for first result (generator): 4.33 seconds
```

`recommendations_stream` is a generator! It yields the extracted products as it's processing the stream in real-time. Now let's get the same response without streaming and see how they compare.

```python
start_perf = time.perf_counter()
recommendations_list = client.create(
    model="gpt-5.4-mini",
    temperature=0.1,
    response_model=Iterable[ProductRecommendation],
    stream=False,
    messages=[
        {
            "role": "system",
            "content": "Generate product recommendations based on the customer profile. Return in order of highest recommended first.",
        },
        {"role": "user", "content": prompt},
    ],
)
print(recommendations_list[0])
end_perf = time.perf_counter()
print(f"Time for first result (list): {end_perf - start_perf:.2f} seconds")
```

```
product_id='1' product_name='Apple MacBook Air (2023)'
Time for first result (list): 8.63 seconds
```

Our web application now displays results faster. Even a 100ms improvement can lead to a 1% increase in revenue.

### FastAPI

We can also take this and set up a streaming LLM API endpoint using FastAPI. Check out our docs on using FastAPI [here](../../concepts/fastapi.md)!

## Key Takeaways

To summarize, we looked at:

• Generators in Python: A powerful feature that allows for efficient data handling with reduced latency

• LLM Streaming: LLMs provide us generators to stream tokens and Instructor can let us validate and extract data from this stream. Real-time data validation ftw!

Don't forget to check our [GitHub](https://github.com/jxnl/instructor) for more resources and give us a star if you find the library helpful!

---

If you have any questions or need further clarifications, feel free to reach out or dive into the Instructor library's documentation for more detailed information. Happy coding!