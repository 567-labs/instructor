---
title: Working with Recursive Schemas in Instructor
description: Learn how to effectively implement and use recursive Pydantic models for handling nested and hierarchical data structures.
---

## See Also

- [Nested Structures](../learning/patterns/nested_structure.md) - Complex hierarchical models
- [Knowledge Graph](./knowledge_graph.md) - Build knowledge graphs
- [Response Models](../concepts/models.md) - Working with complex data structures
- [Types](../concepts/types.md) - Working with different data types

# Recursive Schema Implementation Guide

This guide demonstrates how to work with recursive schemas in Instructor using Pydantic models. While flat schemas are often simpler to work with, some use cases require recursive structures to represent hierarchical data effectively.

!!! tips "Motivation"
    Recursive schemas are particularly useful when dealing with:
    * Nested organizational structures
    * File system hierarchies
    * Comment threads with replies
    * Task dependencies with subtasks
    * Abstract syntax trees

## Defining a Recursive Schema

Here's an example of how to define a recursive Pydantic model:

```python
from typing import List, Optional
from pydantic import BaseModel, Field


class RecursiveNode(BaseModel):
    """A node that can contain child nodes of the same type."""

    name: str = Field(..., description="Name of the node")
    value: Optional[str] = Field(
        None, description="Optional value associated with the node"
    )
    children: List["RecursiveNode"] = Field(
        default_factory=list, description="List of child nodes"
    )


# Required for recursive Pydantic models
RecursiveNode.model_rebuild()
```

## Example Usage

Let's see how to use this recursive schema with Instructor:

```python
import instructor

client = instructor.from_provider("openai/gpt-6-luna")


def parse_hierarchy(text: str) -> RecursiveNode:
    """Parse text into a hierarchical structure."""
    return client.create(
        model="gpt-6-luna",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at parsing text into hierarchical structures.",
            },
            {
                "role": "user",
                "content": f"Parse this text into a hierarchical structure: {text}",
            },
        ],
        response_model=RecursiveNode,
    )


# Example usage
hierarchy = parse_hierarchy("""
Company: Acme Corp
- Department: Engineering
  - Team: Frontend
    - Project: Website Redesign
    - Project: Mobile App
  - Team: Backend
    - Project: API v2
    - Project: Database Migration
- Department: Marketing
  - Team: Digital
    - Project: Social Media Campaign
  - Team: Brand
    - Project: Logo Refresh
""")
```

## Validation and Best Practices

When working with recursive schemas:

1. Always call `model_rebuild()` after defining the model
2. Consider adding validation for maximum depth to prevent infinite recursion
3. Use type hints properly to maintain code clarity
4. Consider implementing custom validators for specific business rules

```python
from pydantic import model_validator


class RecursiveNodeWithDepth(RecursiveNode):
    @model_validator(mode='after')
    def validate_depth(self) -> "RecursiveNodeWithDepth":
        def check_depth(node: "RecursiveNodeWithDepth", current_depth: int = 0) -> int:
            if current_depth > 10:  # Maximum allowed depth
                raise ValueError("Maximum depth exceeded")
            return max(
                [check_depth(child, current_depth + 1) for child in node.children],
                default=current_depth,
            )

        check_depth(self)
        return self
```

## Performance Considerations

While recursive schemas are powerful, they can be more challenging for language models to handle correctly. Consider these tips:

1. Keep structures as shallow as possible
2. Use clear naming conventions
3. Provide good examples in your prompts
4. Consider breaking very large structures into smaller chunks

## Conclusion

Recursive schemas provide a powerful way to handle hierarchical data structures in your applications. While they require more careful handling than flat schemas, they can be invaluable for certain use cases.

For more examples of working with complex data structures, check out:
1. [Query Planning with Dependencies](planning-tasks.md)
2. [Knowledge Graph Generation](knowledge_graph.md)
