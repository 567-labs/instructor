---
title: Generating Synthetic Data with OpenAI's Batch API
description: Learn to use OpenAI's Batch API for large-scale synthetic data generation, focusing on question-answer pairs from the ms-marco dataset.
---

## See Also

- [In-Memory Batch Processing](./batch_in_memory.md) - Serverless batch processing without disk I/O
- [Bulk Classification](./bulk_classification.md) - Process multiple classifications efficiently
- [Cost Optimization](../examples/index.md#api-integration) - Reduce API costs
- [from_provider Guide](../concepts/from_provider.md#async-clients) - Async client setup

# Bulk Generation of Synthetic Data

This tutorial shows how to use `instructor` to generate large quantities of synthetic data at scale using Open AI's new Batch API. In this example, we'll be generating synthetic questions using the `ms-marco` dataset to evaluate RAG retrieval.

??? tips "Why use the batch API?"

    There are a few reasons why you might want to use the Batch API

    1. Batch Jobs are 50% cheaper than running an inference job on demand ( see Open AI's pricing page [here](https://openai.com/api/pricing/) )

    2. Batch Jobs have higher rate limits than normal api calls

    3. Batch Jobs support both normal models **and fine-tuned models**

    This makes them perfect for non time-sensitive tasks that involve large quantities of data.

## Getting Started

Let's first see how we can generate a Question and Answer Pair using Instructor with a normal OpenAI function call.

```python
from pydantic import BaseModel, Field

client = from_openai(OpenAI())


class QuestionAnswerPair(BaseModel):
    """
    This model represents a pair of a question generated from a text chunk, its corresponding answer,
    and the chain of thought leading to the answer. The chain of thought provides insight into how the answer
    was derived from the question.
    """

    chain_of_thought: str = Field(
        description="The reasoning process leading to the answer."
    )
    question: str = Field(description="The generated question from the text chunk.")
    answer: str = Field(description="The answer to the generated question.")


def generate_question(chunk: str) -> QuestionAnswerPair:
    return client.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a world class AI that excels at generating hypothethical search queries. You're about to be given a text snippet and asked to generate a search query which is specific to the specific text chunk that you'll be given. Make sure to use information from the text chunk.",
            },
            {"role": "user", "content": f"Here is the text chunk: {chunk}"},
        ],
        response_model=QuestionAnswerPair,
    )


text_chunk = """
The Reserve Bank of Australia (RBA) came into being on 14 January 1960 as Australia 's central bank and banknote issuing authority, when the Reserve Bank Act 1959 removed the central banking functions from the Commonwealth Bank. The assets of the bank include the gold and foreign exchange reserves of Australia, which is estimated to have a net worth of A$101 billion. Nearly 94% of the RBA's employees work at its headquarters in Sydney, New South Wales and at the Business Resumption Site.
"""
print(generate_question(text_chunk).model_dump_json(indent=2))
"""
{
  "chain_of_thought": "The text discusses the formation of the Reserve Bank of Australia (RBA) and provides key details about its establishment date, the removal of central banking functions from the Commonwealth Bank, its asset worth, and its employee distribution. By focusing on these details, a search query can be framed around the establishment date and purpose of the RBA.",
  "question": "When was the Reserve Bank of Australia established and what are its main functions?",
  "answer": "The Reserve Bank of Australia was established on 14 January 1960 as Australia's central bank and banknote issuing authority."
}
"""
```

As the number of chunks we'd like to generate these synthetic questions for increases, the cost will grow proportionally.

Let's see how we can use the `BatchProcessor` object to create a `.jsonl` file which is compatible with the Batch API.

```python hl_lines="9-18 35-39"
from datasets import load_dataset
from instructor.batch import BatchProcessor
from pydantic import BaseModel, Field
from datasets import load_dataset

dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True).take(200)


def get_messages(dataset):  # (1)!
    for row in dataset:
        for passage in row['passages']['passage_text']:
            yield [
                {
                    "role": "system",
                    "content": "You are a world class AI that excels at generating hypothethical search queries. You're about to be given a text snippet and asked to generate a search query which is specific to the specific text chunk that you'll be given. Make sure to use information from the text chunk.",
                },
                {"role": "user", "content": f"Here is the text chunk: {passage}"},
            ]


class QuestionAnswerPair(BaseModel):
    """
    This model represents a pair of a question generated from a text chunk, its corresponding answer,
    and the chain of thought leading to the answer. The chain of thought provides insight into how the answer
    was derived from the question.
    """

    chain_of_thought: str = Field(
        description="The reasoning process leading to the answer."
    )
    question: str = Field(description="The generated question from the text chunk.")
    answer: str = Field(description="The answer to the generated question.")


processor = BatchProcessor("openai/gpt-4o", QuestionAnswerPair)
processor.create_batch_from_messages(
    messages_list=list(get_messages(dataset)),
    file_path="./test.jsonl",
)  # (2)!
```

1.  We first define a generator which generates a list of messages which we would have made in a normal `openai` api call

2.  We then create a `BatchProcessor` with the model and response_model that we want, and call its `create_batch_from_messages` method. `instructor` will handle the generation of the openai schema behind the scenes as well as write the output to the file path you specify

Once we've got this new `.jsonl` file, we can then use the new `instructor` cli's `batch` command to create a new batch job.

```bash
> % ls -a | grep test.jsonl
test.jsonl

> % instructor batch create-from-file --file-path test.jsonl
```

This will create a table like what you see below. In my case, my batch job took around 6 minutes to complete and cost me $2.72 to run.

| Batch ID                       | Created At          | Status      | Failed | Completed | Total |
| ------------------------------ | ------------------- | ----------- | ------ | --------- | ----- |
| batch_Z8XUudoweH43R9c4sr4wRYub | 2024-07-16 12:45:22 | in_progress | 0      | 483       | 1627  |

Once our batch job is complete, the status will change to `completed`.

??? "Cancelling A Job"

    If you'd like to cancel a batch job midway, you can do so too with the instructor `batch` cli command

    ```bash
    instructor batch cancel --batch-id <batch id here>
    ```

We can then download the file generated by the batch job using the cli command

```bash
instructor batch download-file --download-file-path output.jsonl --batch-id batch_Z8XUudoweH43R9c4sr4wRYub
```

This will then create a `.jsonl` file with the generated content at the path that you specify.

## Parsing the generated response

We can then parse the generated response by using the `.parse_from_file` command provided by the `BatchJob` class.

```python hl_lines="19-21"
from instructor.batch import BatchJob
from pydantic import BaseModel, Field

# <%hide%>
with open("./output.jsonl", "w") as f:
    f.write('')
# <%hide%>


class QuestionAnswerPair(BaseModel):
    """
    This model represents a pair of a question generated from a text chunk, its corresponding answer,
    and the chain of thought leading to the answer. The chain of thought provides insight into how the answer
    was derived from the question.
    """

    chain_of_thought: str = Field(
        description="The reasoning process leading to the answer."
    )
    question: str = Field(description="The generated question from the text chunk.")
    answer: str = Field(description="The answer to the generated question.")


parsed, unparsed = BatchJob.parse_from_file(  # (1)!
    file_path="./output.jsonl", response_model=QuestionAnswerPair
)

print(len(parsed))
#> 0
print(len(unparsed))
#> 0

# <%hide%>
import os

if os.path.exists("./output.jsonl"):
    os.remove("./output.jsonl")
# <%hide%>
```

1.  We can then use a generic `Pydantic` schema to parse the generated function calls back

This will then return a list of two elements

- `parsed` is a list of responses that have been successfully parsed into the `QuestionAnswerPair` Base Model class
- `unparsed` is a second list which contains responses which were not able to be parsed into the `QuestionAnswerPair` Base Model class
