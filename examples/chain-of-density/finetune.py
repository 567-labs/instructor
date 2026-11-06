from openai import OpenAI
from chain_of_density import summarize_article
import csv
import logging
import instructor
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)

client = instructor.from_openai(OpenAI())

instructions = instructor.Instructions(
    name="Chain Of Density",
    finetune_format="messages",
    # log handler is used to save the data to a file
    # you can imagine saving it to a database or other storage
    # based on your needs!
    log_handlers=[logging.FileHandler("generated.jsonl")],
    openai_client=client,
)


class GeneratedSummary(BaseModel):
    """
    This represents a highly concise summary that includes as many entities as possible from the original source article.

    An Entity is a real-world object that's assigned a name - for example, a person, country a product or a book title.

    Guidelines
    - Make every word count
    - The new summary should be highly dense and concise yet self-contained, eg., easily understood without the Article.
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses"
    """

    summary: str = Field(
        ...,
        description="This represents the final summary generated that captures the meaning of the original article which is as concise as possible. ",
    )


@instructions.distil
def distil_summarization(text: str) -> GeneratedSummary:
    summary_chain: list[str] = summarize_article(text)
    return GeneratedSummary(summary=summary_chain[-1])


with open("test.csv") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for article, _summary in reader:
        distil_summarization(article)
