---
authors:
- ivanleomk
categories:
- LLM Techniques
comments: true
date: 2024-09-26
description: Discover how response models impact LLM performance, focusing on structured
  outputs for optimal results in GPT-4o and Claude models.
draft: false
tags:
- LLM Performance
- Response Models
- Structured Outputs
- GPT-4o
- Claude Models
---

# Bad Schemas could break your LLM Structured Outputs

You might be leaving up to 60% performance gains on the table with the wrong response model. Response Models impact model performance massively with Claude and GPT-4o, irregardless of you’re using JSON mode or Tool Calling.

Using the right response model can help ensure [your models respond in the right language](../posts/matching-language.md) or prevent [hallucinations when extracting video timestamps](../posts/timestamp.md).

We decided to investigate this by benchmarking Claude and GPT-4o on the GSM8k dataset and found that

1. **Field Naming drastically impacts performance** - Changing a single field name from `final_choice` to `answer` improved model accuracy from 4.5% to 95%. The way we structure and name fields in our response models can fundamentally alter how the model interprets and responds to queries.
2. **Chain Of Thought significantly boosts performance** - Adding a `reasoning` field increased model accuracy by 60% on the GSM8k dataset. Models perform significantly better when they explain their logic step-by-step.
3. **Be careful with JSON mode** - JSON mode exhibited 50% more performance variation than Tool Calling when renaming fields. Different response models showed varying levels of performance between JSON mode and Tool Calling, indicating that JSON mode requires more careful optimisation.

<!-- more -->

We’ll do so in the following steps

1. We’ll first talk about the GSM8k dataset and how we’re using it for benchmarking
2. Then we’ll cover some of the results we obtained and talk about some of the key takeaways that we discovered
3. Lastly, we’ll provide some tips to optimise your model’s response format that you can apply today

## Dataset

We used OpenAI's GSM8k dataset to benchmark model performance. This dataset challenges LLM models to solve simple math problems that involve multiple steps of reasoning. Here's an example:

> Natalia sold clips to 48 friends in April, and half as many in May. How many clips did Natalia sell in total?"

The original dataset includes reasoning steps and the final answer. We stripped it down to bare essentials: question, answer, and separated reasoning. To do so, we used this code to process the data:

```python
from datasets import load_dataset, Dataset, DatasetDict

splits = ["test", "train"]


def generate_gsm8k(split):
    ds = load_dataset("gsm8k", "main", split=split, streaming=True)
    for row in ds:
        reasoning, answer = row["answer"].split("####")
        answer = int(answer.strip().replace(",", ""))
        yield {
            "question": row["question"],
            "answer": answer,
            "reasoning": reasoning,
        }


# Create the dataset for train and test splits
train_dataset = Dataset.from_generator(lambda: generate_gsm8k("train"))
test_dataset = Dataset.from_generator(lambda: generate_gsm8k("test"))

# Combine them into a DatasetDict
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

dataset.push_to_hub("567-labs/gsm8k")
```

This allows us to test how changes in the response format, response model and even the chosen model itself would affect reasoning ability of the model.

Using this new dataset, we then tested the Claude and GPT-4o models with a variety of different response models and response modes such as JSON Mode and Tool Calling. The final results were fascinating - highlighting the importance of a good response model in squeezing out the maximum performance from your chosen model.

## Benchmarks

We had two key questions on hand that we wanted to answer

1. How does Structured Extraction impact model performance as compared to other response modes such as JSON mode.
2. What was the impact of different response models on model performance?

To answer these questions, we sampled the first 200 questions from the GSM8k dataset and tested different permutations of response modes and response models.

We conducted our experiment in two parts

1. **Modes and Models** : We first started by exploring how different combinations of response modes and models might impact performance on the GSM8k
2. **Response Models :** We then looked at how different response models with varying levels of complexity might impact the performance of each model

Let’s explore each portion in greater detail.

### Modes and Models

By the end of these experiments, we had the following takeaways

1. **Claude Models excel at complex tasks** : Claude models see significantly greater improvement with few shot improvements as compared to the GPT-4o variants. This means that for complex tasks with specific nuanced output formats or instructions, Claude models will benefit more from few-shot examples

2. **Structured Extraction doesn’t lose out** : While we see a 1-2% in performance with JSON mode relative to function calling, working with JSON mode is tricky when response models get complicated. Working with smaller models such as Haiku in JSON mode often required parsing out control characters and increasing the number of re-asks. This was in contrast to the consistent performance of structured extraction that returned a consistent schema.

3. **4o Mini should be used carefully** : We found that 4o-mini had much less steerability as compared to Claude models, with few-shot examples something resulting in worse performance.

It’s important here to note that the few shot examples mentioned here only made a difference when the reasoning behind the answer was provided. Without this reasoning example, there wasn’t the same performance improvement observed.

Here were our results for the Claude Family of models

| Model             | Anthropic JSON Mode | JSON w 5 Few Shot | Anthropic Tools | Tools w 5 few shot | Tools w 10 few shot | Benchmarks |
| ----------------- | ------------------- | ----------------- | --------------- | ------------------ | ------------------- | ---------- |
| claude-3.5-sonnet | 97.00               | 98.5              | 96.00           | 98.00%             | 98%                 | 96.4       |
| claude-3-haiku    | 87.50%              | 89%               | 87.44%          | 90.5%              | 90.5%               | 88.9       |
| claude-3-sonnet   | 94.50%              | 91.5              | 91.00%          | 96.50%             | 91.5%               | 92.3       |
| claude-3-opus     | 96.50%              | 98.50%            | 96.50%          | 97.00%             | 97.00%              | 95         |

Here were our results for `4o-mini`

| model                         | gpt-4o-mini | gpt-4o |
| ----------------------------- | ----------- | ------ |
| Structured Outputs            | 95.5        | 91.5%  |
| Structured Outputs 5 Few-Shot | 94.5        | 94.5%  |
| Tool Calling                  | 93.5        | 93.5%  |
| Tool Calling 5 Few Shot       | 93.0        | 95%    |
| Json Mode                     | 94.5        | 95.5   |
| Json Mode 5 Few Shot          | 95.0        | 97%    |

It’s clear here that Claude models consistently show significant improvement with few-shot examples compared to GPT-4o variants. This is in contrast to `4o-mini` which actually showed a decreased in performance for tool calling when provided with simple examples.

### Response Models

With these new results, we then proceeded to examine how response models might impact the performance of our models when it came to function calling. While doing so, we had the following takeaways.

1. **Chain Of Thought** : Chain Of Thought is incredibly important and can boost model performance on the GSM8k by as much as 60% from our benchmarks
2. **JSON mode is much more sensitive than Tool Calling** : In our initial benchmarks, we found that simple changes in the response model such as additional parameters could impact performance by as much as 30% - something which Tool Calling didn’t suffer from.
3. **Naming matters a lot** : The naming of a response parameter is incredibly important. Just going from `potential_final_choice` and `final_choice` to `potential_answers` and `final_answer` improved our final accuracy from 4.5% to 95%.

#### Chain Of Thought

It’s difficult to understate the importance of allowing the model to reason and plan before generating a final response.

In our initial tests , we used the following two models

```python
class Answer(BaseModel):
    chain_of_thought: str
    answer: int


class OnlyAnswer(BaseModel):
    answer: int
```

| Model      | JSON Mode | Tool Calling |
| ---------- | --------- | ------------ |
| Answer     | 92%       | 94%          |
| OnlyAnswer | 33%       | 33.5%        |

These models were tested using the **exact same prompt and questions**. The only thing that differed between them was the addition of a `chain_of_thought` response parameter to allow the model to reason effectively.

We’re not confined to this specific naming convention of `chain_of_thought`, although it does work consistently well. We can show that when we look at the results we obtained when we tested the following response models.

In order to verify this, we took a random sample of 50 questions from the test dataset and looked at the performance of different response models that implemented similar reasoning fields on the GSM8k.

Our conclusion? Simply adding additional fields for the model to reason about its final response improves reasoning all around.

```python
class AssumptionBasedAnswer(BaseModel):
    assumptions: list[str]
    logic_flow: str
    answer: int


class ErrorAwareCalculation(BaseModel):
    key_steps: list[str]
    potential_pitfalls: list[str]
    intermediate_results: list[str]
    answer: int


class AnswerWithIntermediateCalculations(BaseModel):
    assumptions: list[str]
    intermediate_calculations: list[str]
    chain_of_thought: str
    final_answer: int


class AssumptionBasedAnswerWithExtraFields(BaseModel):
    assumptions: list[str]
    logic_flow: str
    important_intermediate_calculations: list[str]
    potential_answers: list[int]
    answer: int


class AnswerWithReasoningAndCalculations(BaseModel):
    chain_of_thought: str
    key_calculations: list[str]
    potential_answers: list[int]
    final_choice: int
```

| Model                                | Accuracy |
| ------------------------------------ | -------- |
| AssumptionBasedAnswer                | 78%      |
| ErrorAwareCalculation                | 92%      |
| Answer With Intermediate Calculation | 90%      |
| AssumptionBasedAnswerWithExtraFields | 90%      |
| AnswerWithReasoningAndCalculations   | 94%      |

So if you’re generating any sort of response, don’t forget to add in a simple reasoning field that allows for this performance boost.

#### JSON mode is incredibly Sensitive

We were curious how this would translate over to the original sample of 200 questions. To do so, we took the original 200 questions that we sampled in our previous experiment and tried to see how JSON mode and Tool Calling performed with other different permutations with `gpt-4o-mini`.

Here were the models that we used

```python
class Answer(BaseModel):
    chain_of_thought: str
    answer: int


class AnswerWithCalculation(BaseModel):
    chain_of_thought: str
    required_calculations: list[str]
    answer: int


class AssumptionBasedAnswer(BaseModel):
    assumptions: list[str]
    logic_flow: str
    answer: int


class ErrorAwareCalculation(BaseModel):
    key_steps: list[str]
    potential_pitfalls: list[str]
    intermediate_results: list[str]
    answer: int


class AnswerWithNecessaryCalculationAndFinalChoice(BaseModel):
    chain_of_thought: str
    necessary_calculations: list[str]
    potential_final_choices: list[str]
    final_choice: int
```

| Model                                        | JSON Mode | Tool Calling |
| -------------------------------------------- | --------- | ------------ |
| Answer                                       | 92%       | 94%          |
| AnswerWithCalculation                        | 86.5%     | 92%          |
| AssumptionBasedAnswer                        | 65%       | 78.5%        |
| ErrorAwareCalculation                        | 92%       | 88.5%        |
| AnswerWithNecessaryCalculationAndFinalChoice | 87.5%     | 95%          |

What’s interesting about these results is that the difference in performance for JSON mode with multiple response models is far greater than that of Tool Calling.

The worst performing response model for JSON mode was `AssumptionBasedAnswer` which scored 65% on the GSM8k while the worst performing response for Tool Calling was `AssumptionBasedAnswer` that scored 78.5% on our benchmarks. This means that the variation in performance for JSON mode was almost 50% larger than that of Tool Calling.

What’s also interesting is that different response models impacted each response mode differently. For Tool Calling, `AnswerWithNecessaryCalculationAndFinalChoice` was the best performing response model while for JSON mode, it was `ErrorAwareCalculation` and `Answer`.

This means that when looking at response models for our applications, we can’t just toggle a different mode and hope that the performance gets a magical boost. We need to have a systematic way of evaluating model performance to find the best balance between different response models that we’re experimenting with.

#### Naming Matters A Lot

We obtained an accuracy of `4.5%` when working with the following response model

```python
class AnswerWithNecessaryCalculationAndFinalChoice(BaseModel):
    chain_of_thought: str
    necessary_calculations: list[str]
    potential_final_choices: list[str]
    final_choice: int
```

This is weird because it doesn’t look all too different from the top performing response model, which achieved an accuracy of `95%` .

```python
class AnswerWithNecessaryCalculationAndFinalChoice(BaseModel):
    chain_of_thought: str
    necessary_calculations: list[str]
    potential_final_answers: list[str]
    answer: int
```

In fact, the only thing that changed was the last two parameters. Upon closer inspection, what was happening was that in the first case, we were generating response objects that looked like this

```python
{
    "chain_of_thought": "In the race, there are a total of 240 Asians. Given that 80 were Japanese, we can calculate the number of Chinese participants by subtracting the number of Japanese from the total number of Asians: 240 - 80 = 160. Now, it is given that there are 60 boys on the Chinese team. Therefore, to find the number of girls on the Chinese team, we subtract the number of boys from the total number of Chinese participants: 160 - 60 = 100 girls. Thus, the number of girls on the Chinese team is 100.",
    "necessary_calculations": [
        "Total Asians = 240",
        "Japanese participants = 80",
        "Chinese participants = Total Asians - Japanese participants = 240 - 80 = 160",
        "Boys in Chinese team = 60",
        "Girls in Chinese team = Chinese participants - Boys in Chinese team = 160 - 60 = 100",
    ],
    "potential_final_choices": ["60", "100", "80", "120"],
    "final_choice": 2,
}
```

This meant that instead of the final answer of 100, our model was generating potential responses it could give and returning the final choice as the index of that answer. Simply renaming our response model here to `potential_final_answers` and `final_answer` resulted in the original result of `95%` again.

```python
{
    "chain_of_thought": "First, we need to determine how many Asians were Chinese. Since there were 240 Asians in total and 80 of them were Japanese, we can find the number of Chinese by subtracting the number of Japanese from the total: 240 - 80 = 160. Now, we know that there are 160 Chinese participants. Given that there were 60 boys on the Chinese team, we can find the number of girls by subtracting the number of boys from the total number of Chinese: 160 - 60 = 100. Therefore, there are 100 girls on the Chinese team.",
    "necessary_calculations": [
        "Total Asians = 240",
        "Number of Japanese = 80",
        "Number of Chinese = 240 - 80 = 160",
        "Number of boys on Chinese team = 60",
        "Number of girls on Chinese team = 160 - 60 = 100",
    ],
    "potential_final_answers": ["100", "60", "80", "40"],
    "answer": 100,
}
```

These are the sort of insights we’d only be able to know by having a strong evaluation set and looking closely at our generated predictions.

## Why Care about the response model?

It’s pretty obvious that different combinations of field names dramatically impact the performance of models. Ultimately It’s not just about adding a single `chain_of_thought` field but also about paying close attention to how models are interpreting the field names.

For instance, instead of asking for just chain_of_thought, we can be much more creative by prompting our model to generate python code, much like the example below.

```python
class Equations(BaseModel):
    chain_of_thought: str
    eval_string: list[str] = Field(
        description="Python code to evaluate to get the final answer. The final answer should be stored in a variable called `answer`."
    )
```

This allows us to combine a LLM’s expressiveness with the performance of a deterministic system, in this case a python interpreter. As we continue to implement more complex systems with these models, the key isn’t going to be just toggling JSON mode and praying for the best. Instead, we need robust evaluation sets for testing the impact of different response models, prompt changes and other permutations.

## Try Instructor Today

`instructor` makes it easy to get structured data from LLMs and is built on top of Pydantic. This makes it an indispensable tool to quickly prototype and find the right response models for your specific application.

To get started with instructor today, check out our [Getting Started](../../index.md) and [Examples](../../examples/index.md) sections that cover various LLM providers and specialised implementations.