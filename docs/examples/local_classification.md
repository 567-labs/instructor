---
title: Classifying Confidential Data with Local AI Models
description: Learn to classify private documents securely using Llama-cpp-python with instructor while maintaining data privacy and local infrastructure.
---

# Leveraging Local Models for Classifying Private Data

In this article, we'll show you how to use Llama-cpp-python with instructor for classification. This is a perfect use-case for users who want to ensure that confidential documents are handled securely without ever leaving your own infrastructure.

## Setup

Let's start by installing the required libraries in your local python environment. This might take a while since we'll need to build and compile `llama-cpp` for your specific environment.

```bash
pip install instructor pydantic
```

Next, we'll install `llama-cpp-python` which is a python package that allows us to use llama-cpp with our python scripts.

For this tutorial, we'll be using `Mistral-7B-Instruct-v0.2-GGUF` by `TheBloke` to do our function calls. This will require around 6GB of RAM and a GPU.

We can install the package by running the following command

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

!!! note "Don't have a GPU?"

    If you don't have a GPU, we recommend using the `Qwen2-0.5B-Instruct` model instead and compiling llama-cpp-python to use `OpenBLAS`. This allows you to run the program using your CPU instead.

    You can compile `llama-cpp-python` with `OpenBLAS` support by running the command

    ```bash
    CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
    ```

## Using `LLama-cpp-python`

Here's an example of how to implement a system for handling confidential document queries using local models:

```python hl_lines="7-12 14-16 43-52"
from llama_cpp import Llama  # type: ignore
import instructor
from pydantic import BaseModel
from enum import Enum
from typing import Optional

llm = Llama.from_pretrained(  # type: ignore
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",  # (1)!
    filename="*Q4_K_M.gguf",
    verbose=False,  # (2)!
    n_gpu_layers=-1,  # (3)!
)

create = instructor.patch(
    create=llm.create_chat_completion_openai_v1,  # type: ignore  # (4)!
)


# Define query types for document-related inquiries
class QueryType(str, Enum):
    DOCUMENT_CONTENT = "document_content"
    LAST_MODIFIED = "last_modified"
    ACCESS_PERMISSIONS = "access_permissions"
    RELATED_DOCUMENTS = "related_documents"


# Define the structure for query responses
class QueryResponse(BaseModel):
    query_type: QueryType
    response: str
    additional_info: Optional[str] = None


def process_confidential_query(query: str) -> QueryResponse:
    prompt = f"""Analyze the following confidential document query and provide an appropriate response:
    Query: {query}

    Determine the type of query (document content, last modified, access permissions, or related documents),
    provide a response, and include a confidence score and any additional relevant information.
    Remember, you're handling confidential data, so be cautious about specific details.
    """

    return create(
        response_model=QueryResponse,  # (5)!
        messages=[
            {
                "role": "system",
                "content": "You are a secure AI assistant trained to handle confidential document queries.",
            },
            {"role": "user", "content": prompt},
        ],
    )


# Sample confidential document queries
confidential_queries = [
    "What are the key findings in the Q4 financial report?",
    "Who last accessed the merger proposal document?",
    "What are the access permissions for the new product roadmap?",
    "Are there any documents related to Project X's budget forecast?",
    "When was the board meeting minutes document last updated?",
]

# Process each query and print the results
for query in confidential_queries:
    response: QueryResponse = process_confidential_query(query)
    print(f"{query} : {response.query_type}")
    """
    #> What are the key findings in the Q4 financial report? : document_content
    #> Who last accessed the merger proposal document? : access_permissions
    #> What are the access permissions for the new product roadmap? : access_permissions
    #> Are there any documents related to Project X's budget forecast? : document_content
    #> When was the board meeting minutes document last updated? : last_modified
    """
```

1. We load in the model from Hugging Face and cache it locally. This makes it quick and easy for us to experiment with different model configurations and types.

2. We can set `verbose` to be `True` to log out all of the output from `llama.cpp`. This helps if you're trying to debug specific issues

3. If you have a GPU with limited memory, set `n_gpu` to a lower number (Eg. 10 ). We've set it here to `-1` so that all of the model layers are loaded on the GPU by default.

4. Now make sure to patch the client with the `create_chat_completion_openai_v1` api which is OpenAI compatible

5. Pass in the response model as a parameter just like any other inference client we support

## Conclusion

`instructor` provides a robust solution for organizations needing to handle confidential document queries locally. By processing these queries on your own hardware, you can leverage advanced AI capabilities while maintaining the highest standards of data privacy and security.

But this goes far beyond just simple confidential documents, using local models unlocks a whole new world of interesting use-cases, fine-tuned specialist models and more!
