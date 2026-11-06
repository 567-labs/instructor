---
title: Building Knowledge Graphs from Text
description: Learn to construct knowledge graphs from textual data using OpenAI's API and Pydantic in this comprehensive tutorial.
---

## See Also

- [Knowledge Graph](./knowledge_graph.md) - Visualize knowledge graphs
- [Entity Resolution](./entity_resolution.md) - Identify and resolve entities
- [Document Segmentation](./document_segmentation.md) - Break down documents for analysis
- [Nested Structures](../learning/patterns/nested_structure.md) - Complex hierarchical models

# Building Knowledge Graphs from Textual Data

In this tutorial, we will explore the process of constructing knowledge graphs from textual data using OpenAI's API and Pydantic. This approach is crucial for efficiently automating the extraction of structured information from unstructured text.

```python
from typing import List
from pydantic import BaseModel, Field
import instructor


class Node(BaseModel):
    id: int
    label: str
    color: str = "blue"  # Default color set to blue


class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"  # Default color for edges


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)


# Patch the OpenAI client to add response_model support
client = instructor.from_provider("openai/gpt-5-nano")


def generate_graph(input_text: str) -> KnowledgeGraph:
    """Generates a knowledge graph from the input text."""
    return client.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Help me understand the following by describing it as a detailed knowledge graph: {input_text}",
            }
        ],
        response_model=KnowledgeGraph,
    )


if __name__ == "__main__":
    input_text = "Jason is Sarah's friend and he is a doctor"
    graph = generate_graph(input_text)
    print(graph.model_dump_json(indent=2))
    """
    {
      "nodes": [
        {
          "id": 1,
          "label": "Jason",
          "color": "blue"
        },
        {
          "id": 2,
          "label": "Sarah",
          "color": "blue"
        },
        {
          "id": 3,
          "label": "Doctor",
          "color": "blue"
        }
      ],
      "edges": [
        {
          "source": 1,
          "target": 2,
          "label": "is a friend of",
          "color": "black"
        },
        {
          "source": 1,
          "target": 3,
          "label": "is a",
          "color": "black"
        }
      ]
    }
    """
```
