---
date: 2025-05-07
authors:
  - jxnl
categories:
  - tutorials
  - anthropic
  - structured-data
---

# Using Anthropic's Web Search with Instructor for Real-Time Data

Anthropic's new web search tool, when combined with Instructor, provides a powerful way to get real-time, structured data from the web. This allows you to build applications that can answer questions and provide information that is up-to-date, going beyond the knowledge cut-off of large language models.

In this post, we'll explore how to use the `web_search` tool with Instructor to fetch the latest information and structure it into a Pydantic model. Even a simple structure can be very effective for clarity and further processing.

<!-- more -->

## How it Works

The web search tool enables Claude models to perform web searches during a generation. When you provide the `web_search` tool in your API request, Claude can decide to use it if the prompt requires information it doesn't have. The API then executes the search, provides the results back to Claude, and Claude can then use this information to generate a response. Importantly, Claude will cite its sources from the search results. You can find more details in the [official Anthropic documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool).

Instructor simplifies this process by allowing you to define a Pydantic model for the desired output structure. When Claude uses the web search tool and formulates an answer, Instructor ensures that the final output conforms to your defined schema.

## Example: Getting the Latest UFC Results

Let's look at a practical example. We want to get the latest UFC fight results.

First, ensure you have `instructor` and `anthropic` installed:

```bash
uv add instructor anthropic
```

Now, let's define our Pydantic model for the response:

```python
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
```

This Response model is straightforward. It gets the model to first generate a list of citations for articles that it referenced before generating it's answer.

This helps to ground its response in the sources it retrieved and provide a higher quality response.

Now, we can make the API call:

```python
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
```

This approach provides a clean way to get the LLM's answer into a defined Pydantic object. The `examples/anthropic-web-tool/run.py` script reflects this implementation.

Expected output (will vary based on real-time web search data):

```
Response:
The latest UFC event was UFC Fight Night: Sandhagen vs Figueiredo held on May 3, 2025, in Des Moines, Iowa. Cory Sandhagen defeated former champion Deiveson Figueiredo by TKO (knee injury) in the main event, while Reinier de Ridder upset previously undefeated prospect Bo Nickal by TKO in the co-main event [1][2]. The next major UFC event is UFC 315 on May 10, featuring a welterweight championship bout between Belal Muhammad and Jack Della Maddalena [3].

Citations:
1: https://www.ufc.com/news/main-card-results-highlights-winner-interviews-ufc-fight-night-sandhagen-vs-figueiredo-wells-fargo-arena-des-moines
2: https://www.mmamania.com/2025/5/4/24423285/ufc-des-moines-results-sooo-about-last-night-sandhagen-vs-figueiredo-espn-mma-bo-nickal
3: https://en.wikipedia.org/wiki/UFC_315
```

## Key Benefits

- **Real-Time Information**: Access the latest data directly from the web.
- **Structured Output**: Even with a simple model, Instructor ensures the output is a Pydantic object, making it easy to work with programmatically.
- **Source Citations**: Claude automatically cites sources, allowing for verification (details in the API response, not shown in this simplified example).
- **Reduced Hallucinations**: By relying on web search for factual, up-to-the-minute data, the likelihood of the LLM providing incorrect or outdated information is reduced.

## Configuring the Web Search Tool

Anthropic provides several options to configure the web search tool:

- `max_uses`: Limit the number of searches Claude can perform in a single request.
- `allowed_domains`: Restrict searches to a list of specific domains.
- `blocked_domains`: Prevent searches on certain domains.
- `user_location`: Localize search results by providing an approximate location (city, region, country, timezone).

For example, to limit searches to 3 and only allow results from `espn.com` and `ufc.com`:

```python
    tools = (
        [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3,
                "allowed_domains": ["espn.com", "ufc.com"],
            }
        ],
    )
```

You cannot use `allowed_domains` and `blocked_domains` in the same request.

## Conclusion

Combining Anthropic's web search tool with Instructor's structured data capabilities opens up exciting possibilities for building dynamic, information-rich applications. Whether you're tracking sports scores, news updates, or market trends, this powerful duo can help you access and organize real-time web data effectively, even with simple Pydantic models.

Check out the example code in `examples/anthropic-web-tool/run.py` to see this implementation, and refer to the [Anthropic web search documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool) for more in-depth information on the tool's capabilities.
