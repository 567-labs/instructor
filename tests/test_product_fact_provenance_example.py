from examples.product_fact_provenance import FactStatus, ProductExtraction


SOURCES = {
    "supplier-sheet:row-7": "Material: 304 stainless steel.",
    "package-photo:ocr": "Capacity: 750 ml.",
}


def test_marks_source_backed_fact_as_verified() -> None:
    extraction = ProductExtraction.model_validate(
        {
            "facts": [
                {
                    "field_name": "material",
                    "value": "304 stainless steel",
                    "evidence": [
                        {
                            "source_id": "supplier-sheet:row-7",
                            "quote": "Material: 304 stainless steel.",
                        }
                    ],
                }
            ]
        },
        context={"sources": SOURCES},
    )

    fact = extraction.facts[0]
    assert fact.status is FactStatus.VERIFIED
    assert fact.confidence == 1.0
    assert fact.evidence[0].verified is True


def test_keeps_unsupported_fact_visible_as_unverified() -> None:
    extraction = ProductExtraction.model_validate(
        {
            "facts": [
                {
                    "field_name": "waterproof_rating",
                    "value": "IPX7",
                    "evidence": [
                        {
                            "source_id": "supplier-sheet:row-7",
                            "quote": "Waterproof rating: IPX7.",
                        }
                    ],
                }
            ]
        },
        context={"sources": SOURCES},
    )

    fact = extraction.facts[0]
    assert fact.status is FactStatus.UNVERIFIED
    assert fact.confidence == 0.0
    assert fact.evidence[0].verified is False


def test_reports_partial_evidence_confidence() -> None:
    extraction = ProductExtraction.model_validate(
        {
            "facts": [
                {
                    "field_name": "capacity",
                    "value": "750 ml",
                    "evidence": [
                        {
                            "source_id": "package-photo:ocr",
                            "quote": "Capacity: 750 ml.",
                        },
                        {
                            "source_id": "missing-source",
                            "quote": "Capacity: 750 ml.",
                        },
                    ],
                }
            ]
        },
        context={"sources": SOURCES},
    )

    fact = extraction.facts[0]
    assert fact.status is FactStatus.UNVERIFIED
    assert fact.confidence == 0.5
    assert [item.verified for item in fact.evidence] == [True, False]


def test_missing_validation_context_cannot_mark_fact_verified() -> None:
    extraction = ProductExtraction.model_validate(
        {
            "facts": [
                {
                    "field_name": "capacity",
                    "value": "750 ml",
                    "evidence": [
                        {
                            "source_id": "package-photo:ocr",
                            "quote": "Capacity: 750 ml.",
                            "verified": True,
                        }
                    ],
                    "status": "verified",
                    "confidence": 1.0,
                }
            ]
        }
    )

    fact = extraction.facts[0]
    assert fact.status is FactStatus.UNVERIFIED
    assert fact.confidence == 0.0
    assert fact.evidence[0].verified is False
