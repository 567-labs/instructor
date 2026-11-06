---
title: "Document Segmentation with LLMs: A Comprehensive Guide"
description: Learn effective document segmentation techniques using Cohere's LLM, enhancing comprehension of complex texts.
---

## See Also

- [Knowledge Graph](./knowledge_graph.md) - Build knowledge graphs from documents
- [Entity Resolution](./entity_resolution.md) - Identify and disambiguate entities
- [List Extraction](../learning/patterns/list_extraction.md) - Extract multiple objects
- [Nested Structures](../learning/patterns/nested_structure.md) - Complex hierarchical models

# Document Segmentation

In this guide, we demonstrate how to do document segmentation using structured output from an LLM. We'll be using [command-a](https://docs.cohere.com/docs/command-a) - one of Cohere's latest LLMs with 256k context length and testing the approach on an article explaining the Transformer architecture. Same approach to document segmentation can be applied to any other domain where we need to break down a complex long document into smaller chunks.

!!! tips "Motivation"
Sometimes we need a way to split the document into meaningful parts that center around a single key concept/idea. Simple length-based / rule-based text-splitters are not reliable enough. Consider the cases where documents contain code snippets or math equations - we don't want to split those on `'\n\n'` or have to write extensive rules for different types of documents. It turns out that LLMs with sufficiently long context length are well suited for this task.

## Defining the Data Structures

First, we need to define a **`Section`** class for each of the document's segments. **`StructuredDocument`** class will then encapsulate a list of these sections.

Note that in order to avoid LLM regenerating the content of each section, we can simply enumerate each line of the input document and then ask LLM to segment it by providing start-end line numbers for each section.

```python
from pydantic import BaseModel, Field
from typing import List


class Section(BaseModel):
    title: str = Field(description="main topic of this section of the document")
    start_index: int = Field(description="line number where the section begins")
    end_index: int = Field(description="line number where the section ends")


class StructuredDocument(BaseModel):
    """obtains meaningful sections, each centered around a single concept/topic"""

    sections: List[Section] = Field(description="a list of sections of the document")
```

## Document Preprocessing

Preprocess the input `document` by prepending each line with its number.

```python
def doc_with_lines(document):
    document_lines = document.split("\n")
    document_with_line_numbers = ""
    line2text = {}
    for i, line in enumerate(document_lines):
        document_with_line_numbers += f"[{i}] {line}\n"
        line2text[i] = line
    return document_with_line_numbers, line2text
```

## Segmentation

Next use a Cohere client to extract `StructuredDocument` from the preprocessed doc.

```python
# <%hide%>
from pydantic import BaseModel, Field
from typing import List


class Section(BaseModel):
    title: str = Field(description="main topic of this section of the document")
    start_index: int = Field(description="line number where the section begins")
    end_index: int = Field(description="line number where the section ends")


class StructuredDocument(BaseModel):
    """obtains meaningful sections, each centered around a single concept/topic"""

    sections: List[Section] = Field(description="a list of sections of the document")


# <%hide%>

import instructor

# Apply the patch to the cohere client
# enables response_model keyword
client = instructor.from_provider("cohere/command-r-plus")


system_prompt = f"""\
You are a world class educator working on organizing your lecture notes.
Read the document below and extract a StructuredDocument object from it where each section of the document is centered around a single concept/topic that can be taught in one lesson.
Each line of the document is marked with its line number in square brackets (e.g. [1], [2], [3], etc). Use the line numbers to indicate section start and end.
"""


def get_structured_document(document_with_line_numbers) -> StructuredDocument:
    return client.create(
        model="command-a-03-2025",
        response_model=StructuredDocument,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": document_with_line_numbers,
            },
        ],
    )  # type: ignore
```

Next, we need to get back the section text based on the start/end indices and our `line2text` dict from the preprocessing step.

```python
def get_sections_text(structured_doc, line2text):
    segments = []
    for s in structured_doc.sections:
        contents = []
        for line_id in range(s.start_index, s.end_index):
            contents.append(line2text.get(line_id, ''))
        segments.append(
            {
                "title": s.title,
                "content": "\n".join(contents),
                "start": s.start_index,
                "end": s.end_index,
            }
        )
    return segments
```

## Example

Here's an example of using these classes and functions to segment a tutorial on Transformers from [Sebastian Raschka](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html). We can use `trafilatura` package to scrape the web page content of the article.

```python
from trafilatura import fetch_url, extract

# <%hide%>
import instructor
from pydantic import BaseModel, Field
from typing import List


def doc_with_lines(document):
    document_lines = document.split("\n")
    document_with_line_numbers = ""
    line2text = {}
    for i, line in enumerate(document_lines):
        document_with_line_numbers += f"[{i}] {line}\n"
        line2text[i] = line
    return document_with_line_numbers, line2text


client = instructor.from_provider("cohere/command-r-plus")


system_prompt = f"""\
You are a world class educator working on organizing your lecture notes.
Read the document below and extract a StructuredDocument object from it where each section of the document is centered around a single concept/topic that can be taught in one lesson.
Each line of the document is marked with its line number in square brackets (e.g. [1], [2], [3], etc). Use the line numbers to indicate section start and end.
"""


class Section(BaseModel):
    title: str = Field(description="main topic of this section of the document")
    start_index: int = Field(description="line number where the section begins")
    end_index: int = Field(description="line number where the section ends")


class StructuredDocument(BaseModel):
    """obtains meaningful sections, each centered around a single concept/topic"""

    sections: List[Section] = Field(description="a list of sections of the document")


def get_structured_document(document_with_line_numbers) -> StructuredDocument:
    return client.create(
        model="command-a-03-2025",
        response_model=StructuredDocument,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": document_with_line_numbers,
            },
        ],
    )  # type: ignore


def get_sections_text(structured_doc, line2text):
    segments = []
    for s in structured_doc.sections:
        contents = []
        for line_id in range(s.start_index, s.end_index):
            contents.append(line2text.get(line_id, ''))
        segments.append(
            {
                "title": s.title,
                "content": "\n".join(contents),
                "start": s.start_index,
                "end": s.end_index,
            }
        )
    return segments


# <%hide%>

url = 'https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html'
downloaded = fetch_url(url)
document = extract(downloaded)


document_with_line_numbers, line2text = doc_with_lines(document)
structured_doc = get_structured_document(document_with_line_numbers)
segments = get_sections_text(structured_doc, line2text)
```

```
print(segments[5]['title'])
"""
Introduction to Multi-Head Attention
"""
print(segments[5]['content'])
"""
Multi-Head Attention
In the very first figure, at the top of this article, we saw that transformers use a module called multi-head attention. How does that relate to the self-attention mechanism (scaled-dot product attention) we walked through above?
In the scaled dot-product attention, the input sequence was transformed using three matrices representing the query, key, and value. These three matrices can be considered as a single attention head in the context of multi-head attention. The figure below summarizes this single attention head we covered previously:
As its name implies, multi-head attention involves multiple such heads, each consisting of query, key, and value matrices. This concept is similar to the use of multiple kernels in convolutional neural networks.
To illustrate this in code, suppose we have 3 attention heads, so we now extend the \(d' \times d\) dimensional weight matrices so \(3 \times d' \times d\):
In:
h = 3
multihead_W_query = torch.nn.Parameter(torch.rand(h, d_q, d))
multihead_W_key = torch.nn.Parameter(torch.rand(h, d_k, d))
multihead_W_value = torch.nn.Parameter(torch.rand(h, d_v, d))
Consequently, each query element is now \(3 \times d_q\) dimensional, where \(d_q=24\) (here, let’s keep the focus on the 3rd element corresponding to index position 2):
In:
multihead_query_2 = multihead_W_query.matmul(x_2)
print(multihead_query_2.shape)
Out:
torch.Size([3, 24])
"""
```
