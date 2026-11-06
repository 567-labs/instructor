from itertools import product
from typing import Literal
from writerai import AsyncWriter

import pytest
import instructor

from pydantic import BaseModel

from ..util import models, modes


class SinglePrediction(BaseModel):
    """
    Correct class label for the given text
    """

    class_label: Literal["spam", "not_spam"]


data = [
    ("I am a spammer", "spam"),
    ("I am not a spammer", "not_spam"),
]


@pytest.mark.parametrize("model, data, mode", product(models, data, modes))
@pytest.mark.asyncio
async def test_classification(
    model: str,
    data: list[tuple[str, Literal["spam", "not_spam"]]],
    mode: instructor.Mode,
):
    client = instructor.from_writer(client=AsyncWriter(), mode=mode)

    input, expected = data
    resp = await client.chat.completions.create(
        model=model,
        response_model=SinglePrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: {input}",
            },
        ],
    )
    assert resp.class_label == expected


class MultiClassPrediction(BaseModel):
    predicted_labels: list[Literal["billing", "general_query", "hardware"]]


data = [
    (
        "I am having trouble with my billing",
        ["billing"],
    ),
    (
        "I am having trouble with my hardware",
        ["hardware"],
    ),
    (
        "I have a general query and a billing issue",
        ["general_query", "billing"],
    ),
]


@pytest.mark.parametrize("model, data, mode", product(models, data, modes))
@pytest.mark.asyncio
async def test_writer_multi_classify(
    model: str,
    data: list[tuple[str, list[Literal["billing", "general_query", "hardware"]]]],
    mode: instructor.Mode,
):
    client = instructor.from_writer(client=AsyncWriter(), mode=mode)

    input, expected = data

    resp = await client.chat.completions.create(
        model=model,
        response_model=MultiClassPrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following support ticket: {input}. "
                f"Apply this or another class only in cases when "
                f"you sure by 100%.",
            },
        ],
    )
    assert set(resp.predicted_labels) == set(expected)
