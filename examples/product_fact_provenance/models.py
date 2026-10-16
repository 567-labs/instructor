"""Conservative, source-backed validation for extracted product facts."""

from collections.abc import Mapping
from enum import Enum
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidationInfo,
    computed_field,
    model_validator,
)


class FactStatus(str, Enum):
    """Whether every evidence item for a product fact was verified locally."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class Evidence(BaseModel):
    """An exact quote and the source document that contains it."""

    source_id: str = Field(description="Stable ID of the source document")
    quote: str = Field(description="Exact quote copied from the source document")
    _verified: bool = PrivateAttr(default=False)

    @computed_field
    @property
    def verified(self) -> bool:
        """Return the result of local source verification."""

        return self._verified


class ProductFact(BaseModel):
    """A product field whose evidence is checked after model generation."""

    field_name: str = Field(description="Product field, such as material or capacity")
    value: Optional[str] = Field(
        default=None,
        description="Extracted value, or null when the source does not support a value",
    )
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Source IDs and exact quotes that support this value",
    )
    _status: FactStatus = PrivateAttr(default=FactStatus.UNVERIFIED)
    _confidence: float = PrivateAttr(default=0.0)

    @computed_field
    @property
    def status(self) -> FactStatus:
        """Return verified only when every evidence item passes local checks."""

        return self._status

    @computed_field
    @property
    def confidence(self) -> float:
        """Return the fraction of evidence items that passed local checks."""

        return self._confidence

    @model_validator(mode="after")
    def verify_evidence(self, info: ValidationInfo) -> "ProductFact":
        """Check source IDs, exact quotes, and value support without another LLM call."""

        sources = (info.context or {}).get("sources")
        if not isinstance(sources, Mapping) or not self.value or not self.evidence:
            return self

        normalized_value = _normalize(self.value)
        verified_count = 0

        for evidence in self.evidence:
            source_text = sources.get(evidence.source_id)
            evidence._verified = (
                isinstance(source_text, str)
                and evidence.quote in source_text
                and normalized_value in _normalize(evidence.quote)
            )
            verified_count += int(evidence._verified)

        self._confidence = verified_count / len(self.evidence)
        if verified_count == len(self.evidence):
            self._status = FactStatus.VERIFIED

        return self


class ProductExtraction(BaseModel):
    """Product facts extracted from one or more source documents."""

    facts: list[ProductFact]


def _normalize(text: str) -> str:
    """Normalize case and whitespace for a conservative value-in-quote check."""

    return " ".join(text.casefold().split())
