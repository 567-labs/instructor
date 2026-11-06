from instructor import AsyncInstructor, Mode, patch
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

# Initialize the Instructor client with prompt caching
client = AsyncInstructor(
    client=AsyncAnthropic(),
    create=patch(
        create=AsyncAnthropic().beta.prompt_caching.messages.create,
        mode=Mode.TOOLS,
    ),
    mode=Mode.TOOLS,
)


class SituatedContext(BaseModel):
    """
    The context to situate the chunk within the document. The situated context should be as long as the original chunk.

    Example:
       - original chunk: "The company's revenue grew by 3% over the previous quarter."
       - situated context: "This chunk is from an SEC filing on ACME corp's performance in Q2 2023; the previous quarter's revenue was $314 million. The company's revenue grew by 3% over the previous quarter."
    """

    situated_context: str = Field(
        ..., description="The situated context of the chunk within the document."
    )


async def situate_context(doc: str, chunk: str) -> SituatedContext:
    response = await client.chat.completions.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                        <document>
                        {{doc}}
                        </document>
                        """,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": """
                        Here is the chunk we want to situate within the whole document
                        <chunk>
                        {{chunk}}
                        </chunk>

                        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
                        Answer only with the succinct context and nothing else.
                        """,
                    },
                ],
            }
        ],
        response_model=SituatedContext,
        context={
            "doc": doc,
            "chunk": chunk,
        },
    )
    return response


def chunking_function(
    doc: str, chunk_size: int = 1000, overlap: int = 200
) -> list[str]:
    """
    Chunk the document into `chunk_size` character segments with `overlap` overlap.
    """
    chunks = []
    start = 0
    while start < len(doc):
        end = start + chunk_size
        chunks.append(doc[start:end])
        start += chunk_size - overlap
    return chunks


import asyncio


async def process_chunk(doc: str, chunk: str) -> dict[str, str]:
    """
    Process a single chunk by situating it within the context of the full document.

    Args:
    doc (str): The full document text
    chunk (str): A chunk of the document

    Returns:
    Dict[str, str]: A dictionary containing the chunk and its situated context
    """
    context = await situate_context(doc, chunk)
    return {"chunk": chunk, "context": context}


async def process(
    doc: str, chunk_size: int = 1000, overlap: int = 200
) -> list[dict[str, str]]:
    """
    Process the document by chunking it and situating each chunk within the context of the full document.
    Uses asyncio.gather for concurrent processing.

    Args:
    doc (str): The full document text

    Returns:
    List[Dict[str, str]]: A list of dictionaries, each containing a chunk and its situated context
    """
    chunks = chunking_function(doc, chunk_size, overlap)
    tasks = [process_chunk(doc, chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    # Example usage
    document = """
    ACME Corporation Financial Report for Fiscal Year 2023

    Executive Summary:
    ACME Corp. has demonstrated exceptional performance in the latest fiscal year, showcasing significant growth across multiple key areas. This report provides a comprehensive overview of our financial achievements, operational successes, and strategic outlook for the future.

    Financial Highlights:
    1. Revenue: The company's revenue experienced a robust growth of 12% compared to the previous fiscal year, reaching an impressive $1.3 billion. This figure represents a substantial 45% increase from three years ago, underscoring our consistent upward trajectory.

    2. Net Income: Our net income for the fiscal year stood at $180 million, marking a 20% increase from the previous year and an even more impressive 60% rise from three years ago. This growth in net income reflects our improved operational efficiency and successful cost management strategies.

    3. Earnings Per Share (EPS): The EPS for the year reached $4.50, showing a notable improvement from $3.75 in the previous year and $2.80 three years ago. This upward trend in EPS demonstrates our commitment to delivering increasing value to our shareholders.

    4. Gross Margin: Our gross margin improved to 45% in the current fiscal year, up from 41% in the previous year and 38% three years ago, indicating enhanced production efficiency and effective pricing strategies.

    5. Operating Cash Flow: We generated a strong operating cash flow of $250 million in the fiscal year, representing a 50% increase from three years ago.

    Operational Highlights:
    1. Product Launch: ACME's innovative XYZ product line, launched at the beginning of the fiscal year, has surpassed all expectations with an impressive 2 million units sold. This successful launch has significantly contributed to our revenue growth and market share expansion.

    2. Market Expansion: In line with our global growth strategy, we have successfully penetrated ten new international markets during this fiscal year. These new markets have already made a substantial impact, contributing to 20% of the year's total revenue. This expansion not only diversifies our revenue streams but also strengthens our global presence.

    3. Cost Optimization: The implementation of our cutting-edge AI-driven supply chain management system has yielded excellent results, leading to a 10% reduction in operational costs over the past three years. This initiative is part of our ongoing commitment to leveraging technology for improved efficiency and profitability.

    4. Customer Satisfaction: Our customer satisfaction score has improved to 95%, up from 85% three years ago, reflecting our dedication to product quality and customer service excellence.

    5. Sustainability Initiatives: We've made significant strides in our sustainability efforts, reducing our carbon footprint by 30% compared to three years ago. This aligns with our long-term goal of becoming a carbon-neutral organization by 2030.

    Research and Development:
    1. R&D Investment: We've increased our R&D spending by 50% compared to three years ago, focusing on next-generation technologies and sustainable product development.

    2. Patent Filings: ACME filed 60 new patents in the fiscal year, bringing our total patent portfolio to over 1000, further solidifying our position as an industry innovator.

    Looking Ahead:
    Based on the strong fiscal year performance and positive market indicators, ACME Corp. is revising its five-year revenue growth forecast from 50% to a more ambitious 70-80%. This upward revision reflects our confidence in the company's growth trajectory and the effectiveness of our strategic initiatives.

    Key focus areas for the coming years include:
    1. Further expansion into emerging markets in Asia, Africa, and South America.
    2. Continued investment in AI and machine learning to drive operational efficiencies.
    3. Launch of our next-generation sustainable product line, scheduled for the upcoming fiscal year.
    4. Enhancement of our digital transformation initiatives to improve customer experience and internal processes.

    The company remains steadfastly committed to innovation, operational excellence, and sustainable practices to drive long-term growth and shareholder value. We are confident that our strategic initiatives, coupled with our strong market position, will enable us to capitalize on emerging opportunities and navigate potential challenges in the global business landscape.
    In conclusion, ACME Corp.'s performance in Q2 2023 has set a solid foundation for continued success. We thank our employees, customers, and shareholders for their ongoing support and look forward to building on this momentum in the quarters to come.
    """

    async def main():
        import time

        start_time = time.time()
        processed_chunks = await process(document, chunk_size=800, overlap=200)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        for i, item in enumerate(processed_chunks):
            print(f"Chunk {i + 1}:")
            print(f"Text: {item['chunk']}...")
            print(f"Context: {item['context']}")
            print()

    asyncio.run(main())
