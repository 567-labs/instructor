import enum
from itertools import product
from writerai import Writer

import pytest
import instructor

from pydantic import BaseModel

from instructor.mode import Mode
from ..util import models, modes


class Labels(str, enum.Enum):
    SPAM = "spam"
    NOT_SPAM = "not_spam"


class SinglePrediction(BaseModel):
    """
    Correct class label for the given text
    """

    class_label: Labels


data = [
    (
        "I am a spammer",
        Labels.SPAM,
    ),
    (
        "I am not a spammer",
        Labels.NOT_SPAM,
    ),
]


@pytest.mark.parametrize("model, data, mode", product(models, data, modes))
def test_writer_classification(
    model: str, data: list[tuple[str, Labels]], mode: instructor.Mode
):
    client = instructor.from_writer(client=Writer(), mode=mode)

    input, expected = data
    resp = client.chat.completions.create(
        model=model,
        response_model=SinglePrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: {input}. "
                f"Apply this or another class only in cases when "
                f"when you are 100% sure.",
            },
        ],
    )
    assert resp.class_label == expected


class MultiLabels(str, enum.Enum):
    BILLING = "billing"
    GENERAL_QUERY = "general_query"
    HARDWARE = "hardware"


class MultiClassPrediction(BaseModel):
    predicted_labels: list[MultiLabels]


data = [
    (
        "I am having trouble with my billing",
        [MultiLabels.BILLING],
    ),
    (
        "I am having trouble with my hardware",
        [MultiLabels.HARDWARE],
    ),
    (
        "I have a general query and a billing issue",
        [MultiLabels.GENERAL_QUERY, MultiLabels.BILLING],
    ),
]


@pytest.mark.parametrize("model, data, mode", product(models, data, modes))
def test_writer_multi_classify(
    model: str, data: list[tuple[str, list[MultiLabels]]], mode: instructor.Mode
):
    client = instructor.from_writer(client=Writer(), mode=mode)

    if (mode, model) in {
        (Mode.JSON, "gpt-3.5-turbo"),
        (Mode.JSON, "gpt-4"),
    }:
        pytest.skip(f"{mode} mode is not supported for {model}, skipping test")

    input, expected = data

    resp = client.chat.completions.create(
        model=model,
        response_model=MultiClassPrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following support ticket: {input} "
                f"Apply this or another class only in cases when "
                f"when you are 100% sure.",
            },
        ],
    )
    assert set(resp.predicted_labels) == set(expected)
