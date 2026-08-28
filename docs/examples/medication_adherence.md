---
title: Extracting Medication Adherence Triage Notes with Claude
description: Use Anthropic's Claude with Instructor to convert a pharmacist's free-text call note into structured medication-adherence fields (taking-as-prescribed flag, barriers, follow-up action).
---

# Medication Adherence Triage with Anthropic Claude

In this guide, we'll walk through how to convert a pharmacist's free-text call note into a structured medication-adherence record using Anthropic's Claude and Pydantic. This pattern is useful for medication-management workflows where a downstream system needs typed fields — not prose — to decide whether a patient gets a follow-up call, a prescriber outreach, or no further action.

The example deliberately stays at the triage layer. It does **not** generate clinical advice and it does **not** produce a full SOAP note; for a four-section SOAP-style extractor see the dedicated [Action Items](action_items.md) recipe and adapt the response model.

!!! warning "Synthetic data only"

    The transcript below is synthetic. Do not feed real protected health information (PHI) into a third-party model endpoint without the appropriate Business Associate Agreement and de-identification controls in place. For HIPAA-covered workflows, consider routing Claude through [AWS Bedrock](../integrations/bedrock.md) under a BAA.

## Defining the Structure

We model one patient call as an `AdherenceReport`, which contains a per-medication `MedicationAdherence`. Each medication carries:

- `name` — the drug as the patient referred to it (verbatim where possible)
- `taking_as_prescribed` — a single boolean the triage queue can sort on
- `barriers` — enum-typed reasons for non-adherence, so downstream analytics doesn't have to free-text-mine the note
- `escalate_to_clinician` — flips true on adverse-event language or dosing confusion
- `next_action` — short, human-readable instruction for the care manager

Enum-typing the `barriers` field is the single most useful design choice here: it forces Claude to map a patient's wording ("the co-pay went up", "I forget when I'm at work") onto a finite set the rest of the system already knows how to handle.

```python
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Barrier(str, Enum):
    cost = "cost"
    side_effect = "side_effect"
    forgetfulness = "forgetfulness"
    confusion_about_instructions = "confusion_about_instructions"
    feeling_better = "feeling_better"
    access = "access"
    other = "other"


class MedicationAdherence(BaseModel):
    """A single medication discussed during the pharmacist's call."""

    name: str = Field(
        description="Drug name as referenced by the patient. Preserve the patient's wording."
    )
    taking_as_prescribed: bool = Field(
        description="True only if the patient confirms they are taking the medication on the prescribed schedule."
    )
    barriers: List[Barrier] = Field(
        default_factory=list,
        description="Reasons for non-adherence. Empty list if taking_as_prescribed is True.",
    )
    escalate_to_clinician: bool = Field(
        description="True if the patient reports an adverse event, dosing confusion, or any safety concern that a pharmacist should review."
    )
    next_action: str = Field(
        description="Short instruction for the care manager. Use 'no action' if no follow-up is needed."
    )


class AdherenceReport(BaseModel):
    """Structured triage record for a single pharmacist call."""

    patient_reference: Optional[str] = Field(
        default=None,
        description="Opaque patient identifier mentioned in the note, e.g. 'patient A'. Do not invent one.",
    )
    medications: List[MedicationAdherence]
    summary: str = Field(
        description="One-sentence plain-language summary the care manager can read at a glance."
    )
```

## Extracting the Triage Record

We use `Mode.TOOLS` so Claude is forced to populate the schema via tool-use rather than free-text JSON. The system prompt anchors the model to the triage role and pins the escalation rule explicitly — that single instruction is the difference between an agent that quietly under-escalates and one that fires `escalate_to_clinician=True` on adverse-event language.

```python
import instructor

# <%hide%>
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Barrier(str, Enum):
    cost = "cost"
    side_effect = "side_effect"
    forgetfulness = "forgetfulness"
    confusion_about_instructions = "confusion_about_instructions"
    feeling_better = "feeling_better"
    access = "access"
    other = "other"


class MedicationAdherence(BaseModel):
    name: str = Field(description="Drug name as referenced by the patient.")
    taking_as_prescribed: bool
    barriers: List[Barrier] = Field(default_factory=list)
    escalate_to_clinician: bool
    next_action: str


class AdherenceReport(BaseModel):
    patient_reference: Optional[str] = None
    medications: List[MedicationAdherence]
    summary: str


# <%hide%>


client = instructor.from_provider(
    "anthropic/claude-4-5-haiku-latest",
    mode=instructor.Mode.TOOLS,
)


SYSTEM_PROMPT = """You are a triage assistant for a medication-management pharmacist.
You receive a free-text call note and return a structured adherence record.

Rules:
- Set taking_as_prescribed=True only if the patient explicitly confirms they are taking the medication on schedule.
- Map non-adherence reasons onto the Barrier enum. Use 'other' only when no other value fits.
- Set escalate_to_clinician=True for any of: reported adverse event, dosing confusion, missed doses leading to symptom return, or any phrase that sounds like a safety concern.
- next_action must be short and actionable (e.g., 'schedule prescriber outreach for dose review'). Use 'no action' when nothing is required.
- Do not invent patient identifiers, drug names, or facts not present in the note.
"""


CALL_NOTE = """
Patient A called back about their two recent prescriptions. They said they are taking
lisinopril 10 mg every morning without issue. For metformin 500 mg twice daily they
admitted they often skip the evening dose because they forget when they get home late
from work; they also mentioned feeling lightheaded twice last week after taking it on
an empty stomach. They asked whether they should keep taking it the same way.
"""


report = client.create(
    max_tokens=1024,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": CALL_NOTE},
    ],
    response_model=AdherenceReport,
)

print(report.model_dump_json(indent=2))
```

Example output:

```json
{
  "patient_reference": "patient A",
  "medications": [
    {
      "name": "lisinopril 10 mg",
      "taking_as_prescribed": true,
      "barriers": [],
      "escalate_to_clinician": false,
      "next_action": "no action"
    },
    {
      "name": "metformin 500 mg",
      "taking_as_prescribed": false,
      "barriers": ["forgetfulness", "side_effect"],
      "escalate_to_clinician": true,
      "next_action": "schedule prescriber outreach to review metformin timing and lightheadedness"
    }
  ],
  "summary": "Lisinopril adherent; metformin evening dose missed with reported lightheadedness — flag for prescriber review."
}
```

## Why the Schema Looks Like This

A couple of decisions worth flagging:

1. **`barriers` is an enum, not a free string.** A downstream care-management system has to filter and aggregate over these. Enum-typing lets the schema (not the prompt) reject `"cost issues"`, `"too expensive"`, and `"copay"` collapsing into three distinct buckets in the data warehouse.
2. **`escalate_to_clinician` is a separate boolean from `taking_as_prescribed=False`.** A patient can be perfectly adherent and still report an adverse event ("I take it every day but it gives me a rash"). Coupling the two would lose that signal.
3. **`patient_reference` is `Optional[str]` with an explicit instruction not to invent one.** Hallucinated identifiers are worse than missing ones in any clinical workflow.
4. **`Mode.TOOLS` over `Mode.JSON`.** Tool-use gives the model a typed schema target it can validate against on its side too, which empirically reduces malformed-output retries on long notes.

## Re-prompting on Validation Errors

If Claude returns an `AdherenceReport` that doesn't validate (for example, a `barriers` value not in the enum), Instructor will automatically re-prompt with the validation error attached. You can bound the retries with `max_retries`:

```python
report = client.create(
    max_tokens=1024,
    max_retries=2,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": CALL_NOTE},
    ],
    response_model=AdherenceReport,
)
```

For a deeper walkthrough of retries and validators, see [Retrying](../concepts/retrying.md) and [Validators](../concepts/reask_validation.md).

## Next Steps

- Wire the same client through [AWS Bedrock](../integrations/bedrock.md) if you need a BAA path to Claude.
- Add a [Pydantic field validator](../concepts/reask_validation.md) on `next_action` to reject vague follow-ups like `"check in later"`.
- Stream the report in real time with `create_partial` (see the [Anthropic integration page](../integrations/anthropic.md#partials)) if a human triage operator is watching the call note arrive.
