---
title: Extract patient-reported medication use for review
description: Use Claude and Pydantic to extract reported medication use, barriers and follow-up requests with evidence from a synthetic call.
---

# Extract patient-reported medication use for review

A pharmacist reviewing a call needs to see what the caller actually said. This example sends a synthetic transcript to Claude and extracts medication mentions, reported adherence, barriers and explicit follow-up requests. The pharmacist reviews the evidence and decides what action to take.

## Source transcript

> Pharmacist: What would you like to review today?
>
> Caller: I missed two doses of Medicine A because the refill cost too much. Could the pharmacist call me about the refill?
>
> Caller: I also take Medicine B.

## Review the extracted evidence

| Medication | Reported status | Evidence |
| --- | --- | --- |
| Medicine A | Missed doses | “I missed two doses of Medicine A” |
| Medicine B | Unclear | “I also take Medicine B.” establishes the mention. The caller gives no adherence status. |

The cost barrier has its own quote, “the refill cost too much”. The follow-up request is the caller’s question, “Could the pharmacist call me about the refill?” A model validator checks every quote against the transcript. It also requires a supporting quote for each stated adherence status.

## Run the example

From a checkout of Instructor, set `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` for the Claude deployment you want to use. The script prints the structured result for the synthetic call above.

```shell
uv run --extra anthropic python -m examples.medication_adherence.run
```

The example passes the transcript in Instructor’s validation context so a fabricated quote fails validation and can be retried. Verbatim matching establishes that the words occur in the transcript. The reviewer checks that they support the extracted status. [Runnable example](https://github.com/567-labs/instructor/blob/main/examples/medication_adherence/run.py) · [Validation context](../concepts/reask_validation.md)
