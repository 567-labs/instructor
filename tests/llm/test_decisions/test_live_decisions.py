"""Opt-in real decisions API contracts; ordinary test runs never call providers."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
import pytest

import instructor
from instructor.decisions import Noul, Score

pytestmark = pytest.mark.llm


class TicketDecision(BaseModel):
    department: Literal["billing", "technical"] = Field(
        description="Which team handles the customer's request?"
    )
    refund: Annotated[float, Noul("Is the customer requesting a refund?")]
    urgency: Annotated[
        float,
        Score(
            "How urgently does the customer need help?",
            levels=["Not urgent", "Somewhat urgent", "Emergency"],
            scale=(0, 10),
        ),
    ]


STATE = {
    "ticket": "Please refund the duplicate charge on my invoice. There is no rush."
}


def assert_live_contract(
    provider: str, result: TicketDecision, raw: dict[str, Any]
) -> None:
    assert isinstance(result, TicketDecision)
    assert result.department == "billing"
    assert result.refund > 0.5
    assert 0 <= result.urgency <= 10
    assert isinstance(raw["model"], str) and raw["model"]
    answers = raw["answers"]
    assert answers["department"]["type"] == "choice"
    assert answers["department"]["choice"] == result.department
    assert answers["refund"]["type"] == "noul"
    assert answers["refund"]["noul"] == result.refund
    assert answers["urgency"]["type"] == "score"
    assert result.urgency == pytest.approx(answers["urgency"]["score"] * 5)
    for name, expected_keys in (
        ("department", {"billing", "technical"}),
        ("urgency", {"0", "1", "2"}),
    ):
        answer = answers[name]
        probabilities = answer["probabilities"]
        assert set(probabilities) == expected_keys
        assert all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
            for value in probabilities.values()
        )
        assert sum(probabilities.values()) == pytest.approx(1, abs=1e-5)
        confidence = answer["confidence"]
        assert isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        assert math.isfinite(confidence) and 0 <= confidence <= 1
    assert set(answers["urgency"]["legend"]) == {"0", "1", "2"}
    for name in ("input_tokens", "output_tokens"):
        value = raw["usage"][name]
        assert isinstance(value, int) and not isinstance(value, bool) and value > 0
    if provider == "openrouter":
        cost = raw["usage"]["cost"]
        assert isinstance(cost, (int, float)) and not isinstance(cost, bool)
        assert math.isfinite(cost) and cost >= 0


def test_live_sync_decisions_contract(decisions_provider):
    provider, model = decisions_provider
    with instructor.from_provider(
        f"{provider}/{model}", mode=instructor.Mode.DECISIONS
    ) as client:
        result, raw = client.create_with_completion(
            response_model=TicketDecision, context=STATE
        )
    assert_live_contract(provider, result, raw)


@pytest.mark.asyncio
async def test_live_async_decisions_contract(decisions_provider):
    provider, model = decisions_provider
    async with instructor.from_provider(
        f"{provider}/{model}", mode=instructor.Mode.DECISIONS, async_client=True
    ) as client:
        result, raw = await client.create_with_completion(
            response_model=TicketDecision, context=STATE
        )
    assert_live_contract(provider, result, raw)
