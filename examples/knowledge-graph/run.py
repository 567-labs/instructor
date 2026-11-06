import instructor

from graphviz import Digraph
from pydantic import BaseModel, Field
from openai import OpenAI


client = instructor.from_openai(OpenAI())


class Node(BaseModel):
    id: int
    label: str
    color: str


class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"


class KnowledgeGraph(BaseModel):
    nodes: list[Node] = Field(..., default_factory=list)
    edges: list[Edge] = Field(..., default_factory=list)


def generate_graph(input) -> KnowledgeGraph:
    return client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "user",
                "content": f"Help me understand following by describing as a detailed knowledge graph: {input}",
            }
        ],
        response_model=KnowledgeGraph,
    )  # type: ignore


def visualize_knowledge_graph(kg: KnowledgeGraph):
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg.nodes:
        dot.node(str(node.id), node.label, color=node.color)

    # Add edges
    for edge in kg.edges:
        dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color)

    # Render the graph
    dot.render("knowledge_graph.gv", view=True)


graph: KnowledgeGraph = generate_graph("Teach me about quantum mechanics")
visualize_knowledge_graph(graph)
