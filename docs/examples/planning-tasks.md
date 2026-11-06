---
title: Query Planning with Instructor - Complex Task Decomposition
description: Plan and execute complex query plans using Instructor. Break down complex questions into sub-questions with dependencies for systematic information gathering.
---

# Planning and Executing a Query Plan

This example demonstrates how to use the OpenAI Function Call ChatCompletion model to plan and execute a query plan in a question-answering system. By breaking down a complex question into smaller sub-questions with defined dependencies using [lists](../concepts/lists.md), the system can systematically gather the necessary information to answer the main question similar to [knowledge graph extraction](../examples/knowledge_graph.md).

!!! tips "Motivation"

    The goal of this example is to showcase how query planning can be used to handle complex questions, facilitate iterative information gathering, automate workflows, and optimize processes. By leveraging the OpenAI Function Call model, you can design and execute a structured plan to find answers effectively.

     **Use Cases:**

    * Complex question answering
    * Iterative information gathering
    * Workflow automation
    * Process optimization

With the OpenAI Function Call model, you can customize the planning process and integrate it into your specific application to meet your unique requirements.

## Defining the Structures

Let's define the necessary Pydantic models to represent the query plan and the queries.

```python
from typing import List, Literal
from pydantic import Field, BaseModel


class Query(BaseModel):
    """Class representing a single question in a query plan."""

    id: int = Field(..., description="Unique id of the query")
    question: str = Field(
        ...,
        description="Question asked using a question answering system",
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="List of sub questions that need to be answered before asking this question",
    )
    node_type: Literal["SINGLE", "MERGE_MULTIPLE_RESPONSES"] = Field(
        default="SINGLE",
        description="Type of question, either a single question or a multi-question merge",
    )


class QueryPlan(BaseModel):
    """Container class representing a tree of questions to ask a question answering system."""

    query_graph: List[Query] = Field(
        ..., description="The query graph representing the plan"
    )

    def _dependencies(self, ids: List[int]) -> List[Query]:
        """Returns the dependencies of a query given their ids."""
        return [q for q in self.query_graph if q.id in ids]
```

!!! warning "Graph Generation"

    Notice that this example produces a flat list of items with dependencies that resemble a graph, while pydantic allows for recursive definitions, it's much easier and less confusing for the model to generate flat schemas rather than recursive schemas. If you want to see a recursive example, see [recursive schemas](recursive.md)

## Planning a Query Plan

Now, let's demonstrate how to plan and execute a query plan using the defined models and the OpenAI API.

```python
import instructor

# <%hide%>
from typing import List, Literal
from pydantic import Field, BaseModel


class Query(BaseModel):
    """Class representing a single question in a query plan."""

    id: int = Field(..., description="Unique id of the query")
    question: str = Field(
        ...,
        description="Question asked using a question answering system",
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="List of sub questions that need to be answered before asking this question",
    )
    node_type: Literal["SINGLE", "MERGE_MULTIPLE_RESPONSES"] = Field(
        default="SINGLE",
        description="Type of question, either a single question or a multi-question merge",
    )


class QueryPlan(BaseModel):
    """Container class representing a tree of questions to ask a question answering system."""

    query_graph: List[Query] = Field(
        ..., description="The query graph representing the plan"
    )

    def _dependencies(self, ids: List[int]) -> List[Query]:
        """Returns the dependencies of a query given their ids."""
        return [q for q in self.query_graph if q.id in ids]


# <%hide%>

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_provider("openai/gpt-5-nano")


def query_planner(question: str) -> QueryPlan:
    PLANNING_MODEL = "gpt-4o-mini"

    messages = [
        {
            "role": "system",
            "content": "You are a world class query planning algorithm capable ofbreaking apart questions into its dependency queries such that the answers can be used to inform the parent question. Do not answer the questions, simply provide a correct compute graph with good specific questions to ask and relevant dependencies. Before you call the function, think step-by-step to get a better understanding of the problem.",
        },
        {
            "role": "user",
            "content": f"Consider: {question}\nGenerate the correct query plan.",
        },
    ]

    root = client.create(
        model=PLANNING_MODEL,
        temperature=0,
        response_model=QueryPlan,
        messages=messages,
        max_tokens=1000,
    )
    return root
```

```
plan = query_planner(
    "What is the difference in populations of Canada and the Jason's home country?"
)
plan.model_dump()
```

!!! warning "No RAG"

    While we build the query plan in this example, we do not propose a method to actually answer the question. You can implement your own answer function that perhaps makes a retrieval and calls openai for retrieval augmented generation. That step would also make use of function calls but goes beyond the scope of this example.

```python
{
    "query_graph": [
        {
            "dependencies": [],
            "id": 1,
            "node_type": "SINGLE",
            "question": "Identify Jason's home country",
        },
        {
            "dependencies": [],
            "id": 2,
            "node_type": "SINGLE",
            "question": "Find the population of Canada",
        },
        {
            "dependencies": [1],
            "id": 3,
            "node_type": "SINGLE",
            "question": "Find the population of Jason's home country",
        },
        {
            "dependencies": [2, 3],
            "id": 4,
            "node_type": "SINGLE",
            "question": "Calculate the difference in populations between Canada and Jasons home country",
        },
    ]
}
```

In the above code, we define a `query_planner` function that takes a question as input and generates a query plan using the OpenAI API.

## Conclusion

In this example, we demonstrated how to use the OpenAI Function Call `ChatCompletion` model to plan a query using a question-answering system. We defined the necessary structures using Pydantic and created a query planner function that generates a structured plan for answering complex questions.

The query planner breaks down the main question into smaller, manageable sub-questions, establishing dependencies between them. This approach allows for a systematic and organized way to tackle multi-step queries.

For more advanced implementations and variations of this concept, you can explore:

1. [Query planning and execution example](https://github.com/jxnl/instructor/blob/main/examples/query_planner_execution/query_planner_execution.py)
2. [Task planning with topological sort](https://github.com/jxnl/instructor/blob/main/examples/task_planner/task_planner_topological_sort.py)

These examples provide additional insights into how you can leverage structured outputs for complex query planning and task management.

Feel free to adapt this code to your specific use cases and explore the possibilities of using OpenAI Function Calls to plan and structure complex workflows in your applications.
