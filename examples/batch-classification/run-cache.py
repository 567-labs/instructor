import instructor
import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from enum import Enum

client = instructor.from_openai(AsyncOpenAI(), mode=instructor.Mode.TOOLS)
sem = asyncio.Semaphore(5)


class QuestionType(Enum):
    CONTACT = "CONTACT"
    TIMELINE_QUERY = "TIMELINE_QUERY"
    DOCUMENT_SEARCH = "DOCUMENT_SEARCH"
    COMPARE_CONTRAST = "COMPARE_CONTRAST"
    EMAIL = "EMAIL"
    PHOTOS = "PHOTOS"
    SUMMARY = "SUMMARY"


# You can add more instructions and examples in the description
# or you can put it in the prompt in `messages=[...]`
class QuestionClassification(BaseModel):
    """
    Predict the type of question that is being asked.
    Here are some tips on how to predict the question type:
    CONTACT: Searches for some contact information.
    TIMELINE_QUERY: "When did something happen?
    DOCUMENT_SEARCH: "Find me a document"
    COMPARE_CONTRAST: "Compare and contrast two things"
    EMAIL: "Find me an email, search for an email"
    PHOTOS: "Find me a photo, search for a photo"
    SUMMARY: "Summarize a large amount of data"
    """

    # If you want only one classification, just change it to
    #   `classification: QuestionType` rather than `classifications: List[QuestionType]``
    chain_of_thought: str = Field(
        ..., description="The chain of thought that led to the classification"
    )
    classification: list[QuestionType] = Field(
        description=f"An accuracy and correct prediction predicted class of question. Only allowed types: {[t.value for t in QuestionType]}, should be used",
    )

    @field_validator("classification", mode="before")
    def validate_classification(cls, v):
        # sometimes the API returns a single value, just make sure it's a list
        if not isinstance(v, list):
            v = [v]
        return v


# Modify the classify function
async def classify(data: str):
    async with sem:  # some simple rate limiting
        return data, await client.chat.completions.create(
            model="gpt-4",
            response_model=QuestionClassification,
            max_retries=2,
            messages=[
                {
                    "role": "user",
                    "content": f"Classify the following question: {data}",
                },
            ],
        )


async def main(questions: list[str]):
    tasks = [classify(question) for question in questions]
    resps = []
    for task in asyncio.as_completed(tasks):
        question, label = await task
        resp = {
            "question": question,
            "classification": [c.value for c in label.classification],
            "chain_of_thought": label.chain_of_thought,
        }
        resps.append(resp)
    return resps


if __name__ == "__main__":
    import asyncio

    questions = [
        "What was that ai app that i saw on the news the other day?",
        "Can you find the trainline booking email?",
        "What was the book I saw on amazon yesturday?",
        "Can you speak german?",
        "Do you have access to the meeting transcripts?",
        "what are the recent sites I visited?",
        "what did I do on Monday?",
        "Tell me about todays meeting and how it relates to the email on Monday",
    ]

    asyncio.run(main(questions))
