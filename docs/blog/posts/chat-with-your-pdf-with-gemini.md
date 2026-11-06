---
authors:
  - ivanleomk
categories:
  - Gemini
  - Document Processing
comments: true
date: 2024-11-11
description: Learn how to use Google's Gemini model with Instructor to process PDFs and extract structured information
draft: false
tags:
  - Gemini
  - Document Processing
  - PDF Analysis
  - Pydantic
  - Python
---

# PDF Processing with Structured Outputs with Gemini

In this post, we'll explore how to use Google's Gemini model with Instructor to analyse the [Gemini 1.5 Pro Paper](https://github.com/google-gemini/generative-ai-python/blob/0e5c5f25fe4ce266791fa2afb20d17dee780ca9e/third_party/test.pdf) and extract a structured summary.

## The Problem

Processing PDFs programmatically has always been painful. The typical approaches all have significant drawbacks:

- **PDF parsing libraries** require complex rules and break easily
- **OCR solutions** are slow and error-prone
- **Specialized PDF APIs** are expensive and require additional integration
- **LLM solutions** often need complex document chunking and embedding pipelines

What if we could just hand a PDF to an LLM and get structured data back? With Gemini's multimodal capabilities and Instructor's structured output handling, we can do exactly that.

## Quick Setup

First, install the required packages:

```bash
pip install "instructor[google-generativeai]"
```

Then, here's all the code you need:

```python
import instructor
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types.file import File
from pydantic import BaseModel
import time

# Initialize the client
client = instructor.from_provider("google/gemini-2.5-flash")


# Define your output structure
class Summary(BaseModel):
    summary: str


# Upload the PDF
file = genai.upload_file("path/to/your.pdf")

# Wait for file to finish processing
while file.state != File.State.ACTIVE:
    time.sleep(1)
    file = genai.get_file(file.name)
    print(f"File is still uploading, state: {file.state}")

print(f"File is now active, state: {file.state}")
print(file)

resp = client.create(
    messages=[
        {"role": "user", "content": ["Summarize the following file", file]},
    ],
    response_model=Summary,
)

print(resp.summary)
```

??? note "Expand to see Raw Results"

    ```bash
    summary="Gemini 1.5 Pro is a highly compute-efficient multimodal mixture-of-experts model capable of recalling and reasoning over fine-grained information from millions of tokens of context, including multiple long documents and hours of video and audio. It achieves near-perfect recall on long-context retrieval tasks across modalities, improves the state-of-the-art in long-document QA, long-video QA and long-context ASR, and matches or surpasses Gemini 1.0 Ultra's state-of-the-art performance across a broad set of benchmarks. Gemini 1.5 Pro is built to handle extremely long contexts; it has the ability to recall and reason over fine-grained information from up to at least 10M tokens. This scale is unprecedented among contemporary large language models (LLMs), and enables the processing of long-form mixed-modality inputs including entire collections of documents, multiple hours of video, and almost five days long of audio. Gemini 1.5 Pro surpasses Gemini 1.0 Pro and performs at a similar level to 1.0 Ultra on a wide array of benchmarks while requiring significantly less compute to train. It can recall information amidst distractor context, and it can learn to translate a new language from a single set of linguistic documentation. With only instructional materials (a 500-page reference grammar, a dictionary, and ≈ 400 extra parallel sentences) all provided in context, Gemini 1.5 Pro is capable of learning to translate from English to Kalamang, a Papuan language with fewer than 200 speakers, and therefore almost no online presence."
    ```

## Benefits

The combination of Gemini and Instructor offers several key advantages over traditional PDF processing approaches:

**Simple Integration** - Unlike traditional approaches that require complex document processing pipelines, chunking strategies, and embedding databases, you can directly process PDFs with just a few lines of code. This dramatically reduces development time and maintenance overhead.

**Structured Output** - Instructor's Pydantic integration ensures you get exactly the data structure you need. The model's outputs are automatically validated and typed, making it easier to build reliable applications. If the extraction fails, Instructor automatically handles the retries for you with support for [custom retry logic using tenacity](../../concepts/retrying.md).

**Multimodal Support** - Gemini's multimodal capabilities mean this same approach works for various file types. You can process images, videos, and audio files all in the same api request. Check out our [multimodal processing guide](./multimodal-gemini.md) to see how we extract structured data from travel videos.

## Conclusion

Working with PDFs doesn't have to be complicated.

By combining Gemini's multimodal capabilities with Instructor's structured output handling, we can transform complex document processing into simple, Pythonic code.

No more wrestling with parsing rules, managing embeddings, or building complex pipelines - just define your data model and let the LLM do the heavy lifting.

## Related Documentation
- [Multimodal Processing](../../concepts/multimodal.md) - Core multimodal concepts

## See Also
- [Gemini Multimodal Features](multimodal-gemini.md) - Full Gemini capabilities
- [PDF Citation Generation](generating-pdf-citations.md) - Extract citations from PDFs
- [RAG and Beyond](rag-and-beyond.md) - Advanced document processing

If you liked this, give `instructor` a try today and see how much easier structured outputs makes working with LLMs become. [Get started with Instructor today!](../../index.md)
