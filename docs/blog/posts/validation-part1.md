---
authors:
- jxnl
- ivanleomk
categories:
- Pydantic
- Data Validation
- Python
comments: true
date: 2023-10-23
description: Explore dynamic, machine learning-driven validation using Python's Pydantic
  and Instructor to enhance software reliability.
draft: false
tags:
- LLM Validation
- Pydantic
- Python
- Machine Learning
- Software Development
---

# Good LLM Validation is Just Good Validation

> What if your validation logic could learn and adapt like a human, but operate at the speed of software? This is the future of validation and it's already here.

Validation is the backbone of reliable software. But traditional methods are static, rule-based, and can't adapt to new challenges. This post looks at how to bring dynamic, machine learning-driven validation into your software stack using Python libraries like `Pydantic` and `Instructor`. We validate these outputs using a validation function which conforms to the structure seen below.

```python
def validation_function(value):
    if condition(value):
        raise ValueError("Value is not valid")
    return mutation(value)
```

<!-- more -->

## What is Instructor?

`Instructor` helps to ensure you get the exact response type you're looking for when using openai's function call api. Once you've defined the `Pydantic` model for your desired response, `Instructor` handles all the complicated logic in-between - from the parsing/validation of the response to the automatic retries for invalid responses. This means that we can build in validators 'for free' and have a clear separation of concerns between the prompt and the code that calls openai.

```python
import instructor  # pip install instructor
from pydantic import BaseModel

# This enables response_model keyword
# from client.chat.completions.create
client = instructor.from_provider("openai/gpt-5-nano")  # (1)!


class UserDetail(BaseModel):
    name: str
    age: int


user: UserDetail = client.create(
    model="gpt-5.4-mini",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ],
    max_retries=3,  # (2)!
)

assert user.name == "Jason"  # (3)!
assert user.age == 25
```

1.  To simplify your work with OpenAI models and streamline the extraction of Pydantic objects from prompts, we
    offer a patching mechanism for the `ChatCompletion` class.

2.  Invalid responses that fail to be validated successfully will trigger up to as many reattempts as you define.

3.  As long as you pass in a `response_model` parameter to the `ChatCompletion` api call, the returned object will always
    be a validated `Pydantic` object.

In this post, we'll explore how to evolve from static, rule-based validation methods to dynamic, machine learning-driven ones. You'll learn to use `Pydantic` and `Instructor` to leverage language models and dive into advanced topics like content moderation, validating chain of thought reasoning, and contextual validation.

Let's examine how these approaches with an example. Imagine that you run a software company that wants to ensure you never serve hateful and racist content. This isn't an easy job since the language around these topics change very quickly and frequently.

## Software 1.0: Introduction to Validations in Pydantic

A simple method could be to compile a list of different words that are often associated with hate speech. For simplicity, let's assume that we've found that the words `Steal` and `Rob` are good predictors of hateful speech from our database. We can modify our validation structure above to accommodate this.

This will throw an error if we pass in a string like `Let's rob the bank!` or `We should steal from the supermarkets`.

Pydantic offers two approaches for this validation: using the `field_validator` decorator or the `Annotated` hints.

### Using `field_validator` decorator

We can use the `field_validator` decorator to define a validator for a field in Pydantic. Here's a quick example of how we might be able to do so.

```python
from pydantic import BaseModel, ValidationError, field_validator


class UserMessage(BaseModel):
    message: str

    @field_validator('message')
    def message_cannot_have_blacklisted_words(cls, v: str) -> str:
        for word in v.split():  # (1)!
            if word.lower() in {'rob', 'steal'}:
                raise ValueError(f"`{word}` was found in the message `{v}`")
        return v


try:
    UserMessage(message="This is a lovely day")
    UserMessage(message="We should go and rob a bank")
except ValidationError as e:
    print(e)
    """
    1 validation error for UserMessage
    message
      Value error, `rob` was found in the message `We should go and rob a bank` [type=value_error, input_value='We should go and rob a bank', input_type=str]
        For further information visit https://errors.pydantic.dev/2.11/v/value_error
    """
```

1.  We split the sentence into its individual words and iterate through each of the words. We then try to see if any of these
    words are in our blacklist which in this case is just `rob` and `steal`

Since the message `This is a lovely day` does not have any blacklisted words, no errors are thrown. However, in the given example above, the validation fails for the message `We should go and rob a bank` due to the presence of the word `rob` and the corresponding error message is displayed.

```
1 validation error for UserMessage
message
  Value error, `rob` was found in the message `We should go and rob a bank` [type=value_error, input_value='We should go and rob a bank', input_type=str]
    For further information visit https://errors.pydantic.dev/2.4/v/value_error
```

### Using `Annotated`

Alternatively, you can use the `Annotated` function to perform the same validation. Here's an example where we utilise the same function we started with.

```python
from pydantic import BaseModel, ValidationError
from typing import Annotated
from pydantic.functional_validators import AfterValidator


def message_cannot_have_blacklisted_words(value: str):
    for word in value.split():
        if word.lower() in {'rob', 'steal'}:
            raise ValueError(f"`{word}` was found in the message `{value}`")
    return value


class UserMessage(BaseModel):
    message: Annotated[str, AfterValidator(message_cannot_have_blacklisted_words)]


try:
    UserMessage(message="This is a lovely day")
    UserMessage(message="We should go and rob a bank")
except ValidationError as e:
    print(e)
    """
    1 validation error for UserMessage
    message
      Value error, `rob` was found in the message `We should go and rob a bank` [type=value_error, input_value='We should go and rob a bank', input_type=str]
        For further information visit https://errors.pydantic.dev/2.11/v/value_error
    """
```

This code snippet achieves the same validation result. If the user message contains any of the words in the blacklist, a `ValueError` is raised and the corresponding error message is displayed.

```
1 validation error for UserMessage
message
  Value error, `rob` was found in the message `We should go and rob a bank` [type=value_error, input_value='We should go and rob a bank', input_type=str]
    For further information visit https://errors.pydantic.dev/2.4/v/value_error
```

Validation is a fundamental concept in software development and remains the same when applied to AI systems. Existing programming concepts should be leveraged when possible instead of introducing new terms and standards. The underlying principles of validation remain unchanged.

Suppose now that we've gotten a new message - `Violence is always acceptable, as long as we silence the witness`. Our original validator wouldn't throw any errors when passed this new message since it uses neither the words `rob` or `steal`. However, it's clear that it is not a message which should be published. How can we ensure that our validation logic can adapt to new challenges?

## Software 3.0: Validation for LLMs or powered by LLMs

Building upon the understanding of simple field validators, let's delve into probabilistic validation in software 3.0, (prompt engineering). We'll introduce an LLM-powered validator called `llm_validator` that uses a statement to verify the value.

We can get around this by using the inbuilt `llm_validator` class from `Instructor`.

```python
from instructor import llm_validator
from pydantic import BaseModel, ValidationError
from typing import Annotated
from pydantic.functional_validators import AfterValidator


class UserMessage(BaseModel):
    message: Annotated[
        str, AfterValidator(llm_validator("don't say objectionable things"))
    ]


try:
    UserMessage(
        message="Violence is always acceptable, as long as we silence the witness"
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for UserMessage
    message
      Assertion failed, The statement promotes violence, which is objectionable. [type=assertion_error, input_value='Violence is always accep... we silence the witness', input_type=str]
        For further information visit https://errors.pydantic.dev/2.6/v/assertion_error
    """
```

This produces the following error message as seen below

```
1 validation error for UserMessage
message
  Assertion failed, The statement promotes violence, which is objectionable. [type=assertion_error, input_value='Violence is always accep... we silence the witness', input_type=str]
    For further information visit https://errors.pydantic.dev/2.4/v/assertion_error
```

The error message is generated by the language model (LLM) rather than the code itself, making it helpful for re-asking the model in a later section. To better understand this approach, let's see how to build an `llm_validator` from scratch.

### Creating Your Own Field Level `llm_validator`

Building your own `llm_validator` can be a valuable exercise to get started with `Instructor` and create custom validators.

Before we continue, let's review the anatomy of a validator:

```python
def validation_function(value):
    if condition(value):
        raise ValueError("Value is not valid")
    return value
```

As we can see, a validator is simply a function that takes in a value and returns a value. If the value is not valid, it raises a `ValueError`. We can represent this using the following structure:

```python
class Validation(BaseModel):
    is_valid: bool = Field(
        ..., description="Whether the value is valid based on the rules"
    )
    error_message: Optional[str] = Field(
        ...,
        description="The error message if the value is not valid, to be used for re-asking the model",
    )
```

Using this structure, we can implement the same logic as before and utilize `Instructor` to generate the validation.

```python
import instructor

# Enables `response_model` and `max_retries` parameters
client = instructor.from_provider("openai/gpt-5-nano")


def validator(v):
    statement = "don't say objectionable things"
    resp = client.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a validator. Determine if the value is valid for the statement. If it is not, explain why.",
            },
            {
                "role": "user",
                "content": f"Does `{v}` follow the rules: {statement}",
            },
        ],
        # this comes from client = instructor.from_provider("openai/gpt-5-nano")
        response_model=Validation,  # (1)!
    )
    if not resp.is_valid:
        raise ValueError(resp.error_message)
    return v
```

1. The new parameter of `response_model` comes from `client = instructor.from_provider("openai/gpt-5-nano")` and does not exist in the original OpenAI SDK. This
   allows us to pass in the `Pydantic` model that we want as a response.

Now we can use this validator in the same way we used the `llm_validator` from `Instructor`.

```python
class UserMessage(BaseModel):
    message: Annotated[str, AfterValidator(validator)]
```

## Writing more complex validations

### Validating Chain of Thought

A popular way of prompting large language models nowadays is known as chain of thought. This involves getting a model to generate reasons and explanations for an answer to a prompt.

We can utilise `Pydantic` and `Instructor` to perform a validation to check if the reasoning is reasonable, given both the answer and the chain of thought. To do this we can't build a field validator since we need to access multiple fields in the model. Instead we can use a model validator.

```python
def validate_chain_of_thought(values):
    chain_of_thought = values["chain_of_thought"]
    answer = values["answer"]
    resp = client.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a validator. Determine if the value is valid for the statement. If it is not, explain why.",
            },
            {
                "role": "user",
                "content": f"Verify that `{answer}` follows the chain of thought: {chain_of_thought}",
            },
        ],
        # this comes from client = instructor.from_provider("openai/gpt-5-nano")
        response_model=Validation,
    )
    if not resp.is_valid:
        raise ValueError(resp.error_message)
    return values
```

We can then take advantage of the `model_validator` decorator to perform a validation on a subset of the model's data.

> We're defining a model validator here which runs before `Pydantic` parses the input into its respective fields. That's why we have a **before** keyword used in the `model_validator` class.

```python
from pydantic import BaseModel, model_validator


class AIResponse(BaseModel):
    chain_of_thought: str
    answer: str

    @model_validator(mode='before')
    @classmethod
    def chain_of_thought_makes_sense(cls, data: Any) -> Any:
        # here we assume data is the dict representation of the model
        # since we use 'before' mode.
        return validate_chain_of_thought(data)
```

Now, when you create a `AIResponse` instance, the `chain_of_thought_makes_sense` validator will be invoked. Here's an example:

```python
try:
    resp = AIResponse(chain_of_thought="1 + 1 = 2", answer="The meaning of life is 42")
except ValidationError as e:
    print(e)
```

If we create a `AIResponse` instance with an answer that does not follow the chain of thought, we will get an error.

```
1 validation error for AIResponse
    Value error, The statement 'The meaning of life is 42' does not follow the chain of thought: 1 + 1 = 2.
    [type=value_error, input_value={'chain_of_thought': '1 +... meaning of life is 42'}, input_type=dict]
```

### Validating Citations From Original Text

Let's see a more concrete example. Let's say that we've asked our model a question about some text source and we want to validate that the generated answer is supported by the source. This would allow us to minimize hallucinations and prevent statements that are not backed by the original text. While we could verify this by looking up the original source manually, a more scalable approach is to use a validator to do this automatically.

We can pass in additional context to our validation functions using the `model_validate` function in `Pydantic` so that our models have more information to work with when performing validation. This context is a normal python dictionary and can be accessed inside the `info` argument in our validator functions.

```python
from pydantic import ValidationInfo, BaseModel, field_validator


class AnswerWithCitation(BaseModel):
    answer: str
    citation: str

    @field_validator('citation')
    @classmethod
    def citation_exists(cls, v: str, info: ValidationInfo):  # (1)!
        context = info.context
        if context:
            context = context.get('text_chunk')
            if v not in context:
                raise ValueError(f"Citation `{v}` not found in text chunks")
        return v
```

1. This `info` object corresponds to the value of `context` that we pass into the `model_validate` function as seen below.

We can then take our original example and test it against our new model

```python
try:
    AnswerWithCitation.model_validate(
        {"answer": "Jason is a cool guy", "citation": "Jason is cool"},
        context={"text_chunk": "Jason is just a guy"},  # (1)!
    )
except ValidationError as e:
    print(e)
```

1. This `context` object is just a normal python dictionary and can take in and store any arbitrary values

This in turn generates the following error since `Jason is cool` does not exist in the text `Jason is just a guy`.

```
1 validation error for AnswerWithCitation
citation
Value error, Citation `Jason is cool` not found in text chunks [type=value_error, input_value='Jason is cool', input_type=str]
    For further information visit https://errors.pydantic.dev/2.4/v/value_error
```

## Putting it all together with `client = instructor.from_provider("openai/gpt-5-nano")`

To pass this context from the `client.chat.completions.create` call, `client = instructor.from_provider("openai/gpt-5-nano")` also passes the `context`, which will be accessible from the `info` argument in the decorated validator functions.

```python
import instructor

# Enables `response_model` and `max_retries` parameters
client = instructor.from_provider("openai/gpt-5-nano")


def answer_question(question: str, text_chunk: str) -> AnswerWithCitation:
    return client.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "user",
                "content": f"Answer the question: {question} with the text chunk: {text_chunk}",
            },
        ],
        response_model=AnswerWithCitation,
        context={"text_chunk": text_chunk},
    )
```

## Error Handling and Re-Asking

Validators can ensure certain properties of the outputs by throwing errors, in an AI system we can use the errors and allow language model to self correct. Then by running `client = instructor.from_provider("openai/gpt-5-nano")` not only do we add `response_model` and `context` it also allows you to use the `max_retries` parameter to specify the number of times to try and self correct.

This approach provides a layer of defense against two types of bad outputs:

1. Pydantic Validation Errors (code or LLM-based)
2. JSON Decoding Errors (when the model returns an incorrect response)

### Define the Response Model with Validators

To keep things simple let's assume we have a model that returns a `UserModel` object. We can define the response model using Pydantic and add a field validator to ensure that the name is in uppercase.

```python
from pydantic import BaseModel, field_validator


class UserModel(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v.upper() != v:
            raise ValueError("Name must be in uppercase.")
        return v
```

This is where the `max_retries` parameter comes in. It allows the model to self correct and retry the prompt using the error message rather than the prompt.

```python
model = client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
    # Powered by client = instructor.from_provider("openai/gpt-5-nano")
    response_model=UserModel,
    max_retries=2,
)

assert model.name == "JASON"
```

In this example, even though there is no code explicitly transforming the name to uppercase, the model is able to correct the output.

## Conclusion

From the simplicity of Pydantic and Instructor to the dynamic validation capabilities of LLMs, the landscape of validation is changing but without needing to introduce new concepts. It's clear that the future of validation is not just about preventing bad data but about allowing llms to understand the data and correcting it.

If you enjoy the content or want to try out `Instructor` please check out the [github](https://github.com/jxnl/instructor) and give us a star!

## Related Documentation
- [Core Validation Concepts](../../concepts/validation.md) - Learn about validation fundamentals
- [Reask Validation](../../concepts/reask_validation.md) - Handle validation failures gracefully

## See Also
- [Semantic Validation with Structured Outputs](semantic-validation-structured-outputs.md) - Next evolution in validation
- [Why Bad Schemas Break LLMs](bad-schemas-could-break-llms.md) - Schema design best practices
- [Pydantic Is Still All You Need](pydantic-is-still-all-you-need.md) - Why Pydantic validation matters
