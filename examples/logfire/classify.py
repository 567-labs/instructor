import enum
from pydantic import BaseModel
from openai import OpenAI
import instructor
import logfire


class Labels(str, enum.Enum):
    """Enumeration for single-label text classification."""

    SPAM = "spam"
    NOT_SPAM = "not_spam"


class SinglePrediction(BaseModel):
    """
    Class for a single class label prediction.
    """

    class_label: Labels


openai_client = OpenAI()
logfire.configure()
logfire.instrument_pydantic(record="all")
logfire.instrument_openai(openai_client)
client = instructor.from_openai(openai_client)


@logfire.instrument("classification", extract_args=True)
def classify(data: str) -> SinglePrediction:
    """Perform single-label classification on the input text."""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=SinglePrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: {data}",
            },
        ],
    )


if __name__ == "__main__":
    emails = [
        "Hello there I'm a Nigerian prince and I want to give you money",
        "Meeting with Thomas has been set at Friday next week",
        "Here are some weekly product updates from our marketing team",
    ]

    for email in emails:
        classify(email)
