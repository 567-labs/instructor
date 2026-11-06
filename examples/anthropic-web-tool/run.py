import instructor
from pydantic import BaseModel


# Noticed thhat we use JSON not TOOLS mode
client = instructor.from_provider(
    "anthropic/claude-3-7-sonnet-latest",
    mode=instructor.Mode.JSON,
    async_client=False,
)


class Citation(BaseModel):
    id: int
    url: str


class Response(BaseModel):
    citations: list[Citation]
    response: str


response_data, completion_details = client.messages.create_with_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes news articles. Your final response should be only contain a single JSON object returned in your final message to the user. Make sure to provide the exact ids for the citations that support the information you provide in the form of inline citations as [1] [2] [3] which correspond to a unique id you generate for a url that you find in the web search tool which is relevant to your final response.",
        },
        {
            "role": "user",
            "content": "What are the latest results for the UFC and who won? Answer this in a concise response that's under 3 sentences.",
        },
    ],
    tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
    response_model=Response,
)

print("Response:")
print(response_data.response)
print("\nCitations:")
for citation in response_data.citations:
    print(f"{citation.id}: {citation.url}")
