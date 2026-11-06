---
authors:
  - jxnl
categories:
  - LLM
  - Pydantic
comments: true
date: 2024-10-17
description: Explore how to use Instructor and Pydantic to create a pairwise LLM judge for evaluating text relevance.
draft: false
tags:
  - LLM
  - Pydantic
  - Instructor
  - Text Relevance
  - AI Evaluation
---

# Building a Pairwise LLM Judge with Instructor and Pydantic

In this blog post, we'll explore how to create a pairwise LLM judge using Instructor and Pydantic. This judge will evaluate the relevance between a question and a piece of text, demonstrating a practical application of structured outputs in language model interactions.

## Introduction

Evaluating text relevance is a common task in natural language processing and information retrieval. By leveraging large language models (LLMs) and structured outputs, we can create a system that judges the similarity or relevance between a question and a given text.

<!-- more -->

## Setting Up the Environment

First, let's set up our environment with the necessary imports:

```python
import instructor

client = instructor.from_provider("openai/gpt-5-nano")
```

Here, we're using the `instructor` library, which integrates seamlessly with OpenAI's API and Pydantic for structured outputs.

## Defining the Judgment Model

We'll use Pydantic to define a `Judgment` model that structures the output of our LLM:

```python
class Judgment(BaseModel):
    thought: str = Field(
        description="The step-by-step reasoning process used to analyze the question and text"
    )
    justification: str = Field(
        description="Explanation for the similarity judgment, detailing key factors that led to the conclusion"
    )
    similarity: bool = Field(
        description="Boolean judgment indicating whether the question and text are similar or relevant (True) or not (False)"
    )
```

This model ensures that our LLM's output is structured and includes a thought process, justification, and a boolean similarity judgment.

## Creating the Judge Function

Next, we'll create a function that uses our LLM to judge the relevance between a question and a text:

```python
def judge_relevance(question: str, text: str) -> Judgment:
    return client.chat.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": """
                    You are tasked with comparing a question and a piece of text to determine if they are relevant to each other or similar in some way. Your goal is to analyze the content, context, and potential connections between the two.

                    To determine if the question and text are relevant or similar, please follow these steps:

                    1. Carefully read and understand both the question and the text.
                    2. Identify the main topic, keywords, and concepts in the question.
                    3. Analyze the text for any mention of these topics, keywords, or concepts.
                    4. Consider any potential indirect connections or implications that might link the question and text.
                    5. Evaluate the overall context and purpose of both the question and the text.

                    As you go through this process, please use a chain of thought approach. Write out your reasoning for each step inside <thought> tags.

                    After your analysis, provide a boolean judgment on whether the question and text are similar or relevant to each other. Use "true" if they are similar or relevant, and "false" if they are not.

                    Before giving your final judgment, provide a justification for your decision. Explain the key factors that led to your conclusion.

                    Please ensure your analysis is thorough, impartial, and based on the content provided.
                """,
            },
            {
                "role": "user",
                "content": """
                    Here is the question:

                    <question>
                    {{question}}
                    </question>

                    Here is the text:
                    <text>
                    {{text}}
                    </text>
                """,
            },
        ],
        response_model=Judgment,
        context={"question": question, "text": text},
    )
```

This function takes a question and a text as input, sends them to the LLM with a predefined prompt, and returns a structured `Judgment` object.

## Testing the Judge

To test our pairwise LLM judge, we can create a set of test pairs and evaluate the judge's performance:

```python
if __name__ == "__main__":
    test_pairs = [
        {
            "question": "What are the main causes of climate change?",
            "text": "Global warming is primarily caused by human activities, such as burning fossil fuels, deforestation, and industrial processes. These activities release greenhouse gases into the atmosphere, trapping heat and leading to a rise in global temperatures.",
            "is_similar": True,
        },
        # ... (other test pairs)
    ]

    score = 0
    for pair in test_pairs:
        result = judge_relevance(pair["question"], pair["text"])
        if result.similarity == pair["is_similar"]:
            score += 1

    print(f"Score: {score}/{len(test_pairs)}")
    #> Score 9/10
```

This test loop runs the judge on each pair and compares the result to a predetermined similarity value, calculating an overall score.

## Conclusion

By combining Instructor, Pydantic, and OpenAI's language models, we've created a powerful tool for judging text relevance. This approach demonstrates the flexibility and power of structured outputs in LLM applications.

The pairwise LLM judge we've built can be used in various scenarios, such as:

1. Improving search relevance in information retrieval systems
2. Evaluating the quality of question-answering systems
3. Assisting in content recommendation algorithms
4. Automating parts of the content moderation process

As you explore this technique, consider how you might extend or adapt it for your specific use cases. The combination of structured outputs and large language models opens up a world of possibilities for creating intelligent, interpretable AI systems.
