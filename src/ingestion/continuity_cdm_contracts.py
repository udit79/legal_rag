"""Provider-facing classification contracts and response normalization."""

import math
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from src.ingestion.continuity_contracts import ChunkReference, ContinuityRelation
from src.ingestion.contracts import NonEmptyStr, NonNegativeInt

_ExactNonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
_Probability = Annotated[
    float,
    Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True),
]
_NonNegativeCost = Annotated[float, Field(ge=0.0, allow_inf_nan=False, strict=True)]
_PROBABILITY_TOLERANCE = 5e-4


class CDMClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: NonEmptyStr
    source: ChunkReference
    target: ChunkReference
    source_text: _ExactNonEmptyStr
    target_text: _ExactNonEmptyStr
    model_id: NonEmptyStr = "classify-1"

    @field_validator("source_text", "target_text")
    @classmethod
    def validate_chunk_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chunk text must not be blank")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        if self.source.document_id != self.target.document_id:
            raise ValueError("Chunks must belong to the same document")
        if self.target.page_number != self.source.page_number + 1:
            raise ValueError("Chunks must belong to adjacent pages")
        if self.target.chunk_index <= self.source.chunk_index:
            raise ValueError("Target chunk index must be greater than source")
        return self


class CDMUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    cost: _NonNegativeCost | None = None


class CDMClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: NonEmptyStr
    relation_type: ContinuityRelation
    probabilities: dict[ContinuityRelation, _Probability]
    model_id: NonEmptyStr
    request_id: str | None = None
    usage: CDMUsage | None = None

    @model_validator(mode="after")
    def validate_probabilities(self) -> Self:
        expected = set(ContinuityRelation)
        if set(self.probabilities) != expected:
            raise ValueError("Probabilities must contain exactly all four relations")
        if any(not math.isfinite(value) for value in self.probabilities.values()):
            raise ValueError("Probability values must be finite")
        if abs(sum(self.probabilities.values()) - 1.0) > _PROBABILITY_TOLERANCE:
            raise ValueError("Probability values must sum to 1 within 1e-4")
        maximum = max(self.probabilities.values())
        if self.probabilities[self.relation_type] != maximum:
            raise ValueError("Selected relation must have a highest-scoring probability")
        return self


def adapt_scaledown_response(
    response: Any,
    request: CDMClassificationRequest,
) -> CDMClassificationResult:
    """Normalize the observed Decisions API response into the internal result."""
    try:
        answer = response["answers"]["continuity_relation"]
        if answer.get("type") != "choice":
            raise ValueError("Expected a choice answer from ScaleDown")
        raw_probabilities = answer["probabilities"]
        usage = response.get("usage")
        model_id = response["model"]
        if model_id != request.model_id:
            raise ValueError("ScaleDown returned an unexpected model")
        return CDMClassificationResult(
            candidate_id=request.candidate_id,
            relation_type=answer["choice"],
            probabilities=raw_probabilities,
            model_id=model_id,
            request_id=response.get("id") or response.get("request_id"),
            usage=usage,
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("Malformed ScaleDown classification response") from exc
