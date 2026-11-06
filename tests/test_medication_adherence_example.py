import pytest
from pydantic import ValidationError

from examples.medication_adherence.run import AdherenceReport, SAMPLE_TRANSCRIPT


def medication(**updates):
    return {
        "name": "Medicine A",
        "mention_quote": "I missed two doses of Medicine A because the refill cost too much.",
        "reported_status": "missed_doses",
        "status_quote": "I missed two doses of Medicine A",
        "barriers": [{"category": "cost", "quote": "the refill cost too much"}],
        **updates,
    }


def validate(meds, requests=()):
    return AdherenceReport.model_validate(
        {"medications": meds, "follow_up_requests": list(requests)},
        context={"transcript": SAMPLE_TRANSCRIPT},
    )


def test_source_quotes_and_explicit_follow_up_request():
    report = validate(
        [
            medication(),
            {
                "name": "Medicine B",
                "mention_quote": "I also take Medicine B.",
                "reported_status": "unclear",
            },
        ],
        ["Could the pharmacist call me about the refill?"],
    )
    assert report.medications[0].barriers[0].category == "cost"
    assert report.medications[1].status_quote is None


@pytest.mark.parametrize(
    "updates",
    [
        {"name": "Medicine C"},
        {"status_quote": None},
        {"status_quote": "I missed three doses"},
        {"barriers": [{"category": "cost", "quote": "I could not pay for it"}]},
    ],
)
def test_invented_or_missing_evidence_is_rejected(updates):
    with pytest.raises(ValidationError):
        validate([medication(**updates)])


def test_unrequested_follow_up_is_rejected():
    with pytest.raises(ValidationError, match="verbatim"):
        validate([medication()], ["Change the prescribed dose."])
