import enum
import instructor
from openai import OpenAI

from pydantic import BaseModel

client = instructor.from_openai(OpenAI())


class Labels(str, enum.Enum):
    SPAM = "spam"
    NOT_SPAM = "not_spam"


class SinglePrediction(BaseModel):
    """
    Correct class label for the given text
    """

    class_label: Labels


def classify(data: str) -> SinglePrediction:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=SinglePrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: {data}",
            },
        ],
    )  # type: ignore


prediction = classify("Hello there I'm a nigerian prince and I want to give you money")
assert prediction.class_label == Labels.SPAM
