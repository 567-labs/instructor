from openai import OpenAI
import instructor

from graphviz import Digraph
from typing import Optional

from pydantic import BaseModel, Field

client = instructor.from_openai(OpenAI())


class Node(BaseModel):
    id: int
    label: str
    color: str

    def __hash__(self) -> int:
        return hash((id, self.label))


class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.label))


class KnowledgeGraph(BaseModel):
    nodes: Optional[list[Node]] = Field(..., default_factory=list)
    edges: Optional[list[Edge]] = Field(..., default_factory=list)

    def update(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Updates the current graph with the other graph, deduplicating nodes and edges."""
        return KnowledgeGraph(
            nodes=list(set(self.nodes + other.nodes)),
            edges=list(set(self.edges + other.edges)),
        )

    def draw(self, prefix: str = None):
        dot = Digraph(comment="Knowledge Graph")

        # Add nodes
        for node in self.nodes:
            dot.node(str(node.id), node.label, color=node.color)

        # Add edges
        for edge in self.edges:
            dot.edge(
                str(edge.source), str(edge.target), label=edge.label, color=edge.color
            )
        dot.render(prefix, format="png", view=True)


def generate_graph(input: list[str]) -> KnowledgeGraph:
    cur_state = KnowledgeGraph()
    num_iterations = len(input)
    for i, inp in enumerate(input):
        new_updates = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": """You are an iterative knowledge graph builder.
                    You are given the current state of the graph, and you must append the nodes and edges 
                    to it Do not procide any duplcates and try to reuse nodes as much as possible.""",
                },
                {
                    "role": "user",
                    "content": f"""Extract any new nodes and edges from the following:
                    # Part {i}/{num_iterations} of the input:

                    {inp}""",
                },
                {
                    "role": "user",
                    "content": f"""Here is the current state of the graph:
                    {cur_state.model_dump_json(indent=2)}""",
                },
            ],
            response_model=KnowledgeGraph,
        )  # type: ignore

        # Update the current state
        cur_state = cur_state.update(new_updates)
        cur_state.draw(prefix=f"iteration_{i}")
    return cur_state


# here we assume that we have to process the text in chunks
# one at a time since they may not fit in the prompt otherwise
text_chunks = [
    "Jason knows a lot about quantum mechanics. He is a physicist. He is a professor",
    "Professors are smart.",
    "Sarah knows Jason and is a student of his.",
    "Sarah is a student at the University of Toronto. and UofT is in Canada.",
]

graph: KnowledgeGraph = generate_graph(text_chunks)

graph.draw(prefix="final")
