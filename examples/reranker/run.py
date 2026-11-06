import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator, ValidationInfo

# Initialize the OpenAI client with Instructor
client = instructor.from_openai(OpenAI())


class Label(BaseModel):
    chunk_id: str = Field(description="The unique identifier of the text chunk")
    chain_of_thought: str = Field(
        description="The reasoning process used to evaluate the relevance"
    )
    relevancy: int = Field(
        description="Relevancy score from 0 to 10, where 10 is most relevant",
        ge=0,
        le=10,
    )

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, v: str, info: ValidationInfo) -> str:
        context = info.context
        chunks = context.get("chunks", [])
        if v not in [chunk["id"] for chunk in chunks]:
            raise ValueError(
                f"Chunk with id {v} not found, must be one of {[chunk['id'] for chunk in chunks]}"
            )
        return v


class RerankedResults(BaseModel):
    labels: list[Label] = Field(description="List of labeled and ranked chunks")

    @field_validator("labels")
    @classmethod
    def model_validate(cls, v: list[Label]) -> list[Label]:
        return sorted(v, key=lambda x: x.relevancy, reverse=True)


def rerank_results(query: str, chunks: list[dict]) -> RerankedResults:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=RerankedResults,
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert search result ranker. Your task is to evaluate the relevance of each text chunk to the given query and assign a relevancy score.

                For each chunk:
                1. Analyze its content in relation to the query.
                2. Provide a chain of thought explaining your reasoning.
                3. Assign a relevancy score from 0 to 10, where 10 is most relevant.

                Be objective and consistent in your evaluations.
                """,
            },
            {
                "role": "user",
                "content": """
                <query>{{ query }}</query>

                <chunks_to_rank>
                {% for chunk in chunks %}
                <chunk chunk_id="{{ chunk.id }}">
                    {{ chunk.text }}
                </chunk>
                {% endfor %}
                </chunks_to_rank>

                Please provide a RerankedResults object with a Label for each chunk.
                """,
            },
        ],
        context={"query": query, "chunks": chunks},
    )


def main():
    # Sample query and chunks
    query = "What are the health benefits of regular exercise?"
    chunks = [
        {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "text": "Regular exercise can improve cardiovascular health and reduce the risk of heart disease.",
        },
        {
            "id": "b2c3d4e5-f6g7-8901-bcde-fg2345678901",
            "text": "The price of gym memberships varies widely depending on location and facilities.",
        },
        {
            "id": "c3d4e5f6-g7h8-9012-cdef-gh3456789012",
            "text": "Exercise has been shown to boost mood and reduce symptoms of depression and anxiety.",
        },
        {
            "id": "d4e5f6g7-h8i9-0123-defg-hi4567890123",
            "text": "Proper nutrition is essential for maintaining a healthy lifestyle.",
        },
        {
            "id": "e5f6g7h8-i9j0-1234-efgh-ij5678901234",
            "text": "Strength training can increase muscle mass and improve bone density, especially important as we age.",
        },
    ]

    # Rerank the results
    results = rerank_results(query, chunks)

    # Print the reranked results
    print("Reranked results:")
    for label in results.labels:
        print(f"Chunk {label.chunk_id} (Relevancy: {label.relevancy}):")
        print(
            f"Text: {next(chunk['text'] for chunk in chunks if chunk['id'] == label.chunk_id)}"
        )
        print(f"Reasoning: {label.chain_of_thought}")
        print()


if __name__ == "__main__":
    main()
