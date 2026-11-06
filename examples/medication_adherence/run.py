"""Extract patient-reported medication statements for pharmacist review."""

import os
from typing import Literal

import instructor
from pydantic import BaseModel, Field, ValidationInfo, model_validator


SAMPLE_TRANSCRIPT = (
    "Pharmacist: What would you like to review today?\n"
    "Caller: I missed two doses of Medicine A because the refill cost too much. "
    "Could the pharmacist call me about the refill?\n"
    "Caller: I also take Medicine B."
)


class ReportedBarrier(BaseModel):
    category: Literal[
        "cost", "side_effect", "forgetfulness", "instructions", "access", "other"
    ]
    quote: str = Field(
        description="Exact words the caller used to describe the barrier."
    )


class MedicationReport(BaseModel):
    name: str = Field(description="Medication name in the caller's own words.")
    mention_quote: str
    reported_status: Literal["as_prescribed", "missed_doses", "unclear"]
    status_quote: str | None = None
    barriers: list[ReportedBarrier] = Field(default_factory=list)


class AdherenceReport(BaseModel):
    medications: list[MedicationReport]
    follow_up_requests: list[str] = Field(
        default_factory=list,
        description="Exact requests the caller made for follow-up.",
    )

    @model_validator(mode="after")
    def check_evidence(self, info: ValidationInfo):
        transcript = (info.context or {}).get("transcript")
        if not isinstance(transcript, str):
            raise ValueError("Supply the source transcript as validation context")
        quotes = list(self.follow_up_requests)
        for medication in self.medications:
            if (
                not medication.name.strip()
                or medication.name not in medication.mention_quote
            ):
                raise ValueError("The medication name must appear in its source quote")
            quotes.append(medication.mention_quote)
            if medication.reported_status != "unclear" and not medication.status_quote:
                raise ValueError("A reported adherence status requires a source quote")
            if medication.status_quote is not None:
                quotes.append(medication.status_quote)
            quotes.extend(barrier.quote for barrier in medication.barriers)
        if any(not quote.strip() or quote not in transcript for quote in quotes):
            raise ValueError(
                "Every quote must occur verbatim in the supplied transcript"
            )
        return self


def extract_report(transcript: str, model: str) -> AdherenceReport:
    client = instructor.from_provider(f"anthropic/{model}")
    return client.create(
        response_model=AdherenceReport,
        context={"transcript": transcript},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract only the caller's reported medication use, barriers and explicit "
                    "requests. Copy the evidence verbatim. Use unclear when the caller does not "
                    "say whether they took the medicine as prescribed. Leave clinical decisions "
                    "and follow-up actions to the pharmacist."
                ),
            },
            {"role": "user", "content": transcript},
        ],
    )


if __name__ == "__main__":
    report = extract_report(SAMPLE_TRANSCRIPT, os.environ["ANTHROPIC_MODEL"])
    print(report.model_dump_json(indent=2))
