---
description: "Automate few-shot chain of thought to choose diverse examples"
---

How can we improve the performance of few-shot CoT?

While few-shot CoT reasoning is effective, its effectiveness relies on manually crafted examples. Further, choosing diverse examples has shown effective in reducing reasoning errors from CoT.

Here, we automate CoT to choose diverse examples. Given a list of potential examples:

1. **Cluster**: Cluster potential examples
2. **Sample**: For each cluster,
   1. Sort examples by distance from cluster center
   2. Select the first example that meets a predefined selection criteria
3. **Prompt**: Incorporate the chosen questions from each cluster as examples in the LLM prompt

!!! info

    A sample selection criteria could be limiting the number of reasoning steps to a maximum of 5 steps to encourage sampling examples with simpler rationales.

```python hl_lines="72 75 106"
import instructor
import numpy as np
from openai import OpenAI
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

client = instructor.from_provider("openai/gpt-4o")
NUM_CLUSTERS = 2


class Example(BaseModel):
    question: str
    reasoning_steps: list[str]


class FinalAnswer(BaseModel):
    reasoning_steps: list[str]
    answer: int


def cluster_and_sort(questions, n_clusters=NUM_CLUSTERS):
    # Cluster
    embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(questions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embeddings)

    # Sort
    sorted_clusters = [[] for _ in range(kmeans.n_clusters)]
    for question, embedding, label in zip(questions, embeddings, kmeans.labels_):
        center = kmeans.cluster_centers_[label]
        distance = np.linalg.norm(embedding - center)
        sorted_clusters[label].append((distance, question))
    for cluster in sorted_clusters:
        cluster.sort()  # Sort by distance

    return sorted_clusters


def sample(cluster):
    for question in cluster:
        response = client.create(
            model="gpt-4o",
            response_model=Example,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that generates step-by-step reasoning for mathematical questions.",
                },
                {
                    "role": "user",
                    "content": f"Q: {question}\nA: Let's think step by step.",
                },
            ],
        )
        if (
            len(response.reasoning_steps) <= 5
        ):  # If we satisfy the selection criteria, we've found our question for this cluster
            return response


if __name__ == "__main__":
    questions = [
        "How many apples are left if you have 10 apples and eat 3?",
        "What's the sum of 5 and 7?",
        "If you have 15 candies and give 6 to your friend, how many do you have left?",
        "What's 8 plus 4?",
        "You start with 20 stickers and use 8. How many stickers remain?",
        "Calculate 6 added to 9.",
    ]

    # Cluster and sort the questions
    sorted_clusters = cluster_and_sort(questions)

    # Sample questions that match selection criteria for each cluster
    selected_examples = [sample(cluster) for cluster in sorted_clusters]
    print(selected_examples)
    """
    [
        Example(
            question='If you have 15 candies and give 6 to your friend, how many do you have left?',
            reasoning_steps=[
                'Start with the total number of candies you have, which is 15.',
                'Subtract the number of candies you give to your friend, which is 6, from the total candies.',
                '15 - 6 = 9, so you are left with 9 candies.',
            ],
        ),
        Example(
            question="What's the sum of 5 and 7?",
            reasoning_steps=[
                'Identify the numbers to be added: 5 and 7.',
                'Perform the addition: 5 + 7.',
                'The sum is 12.',
            ],
        ),
    ]
    """

    # Use selected questions as examples for the LLM
    response = client.create(
        model="gpt-4o",
        response_model=FinalAnswer,
        messages=[
            {
                "role": "user",
                "content": f"""
                {selected_examples}
                If there are 10 books in my bad and I read 8 of them, how many books do I have left? Let's think step by step.
                """,
            }
        ],
    )

    print(response.reasoning_steps)
    """
    [
        'Start with the total number of books in the bag, which is 10.',
        "Subtract the number of books you've read, which is 8, from the total books.",
        '10 - 8 = 2, so you have 2 books left.',
    ]
    """
    print(response.answer)
    #> 2
```

### References

<sup id="ref-1">1</sup>: [Automatic Chain of Thought Prompting in Large Language Models](https://arxiv.org/abs/2210.03493)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
