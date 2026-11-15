"""Non-adjudicative knowledge-assertion records and their classifications."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AssertionClassification(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    QUALIFIES = "QUALIFIES"
    SUPERSEDES = "SUPERSEDES"
    UNRESOLVED = "UNRESOLVED"


class AssertionContext(BaseModel):
    """Explicit applicability metadata; absent values remain unknown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    jurisdiction: str | None = None
    scope: str | None = None
    valid_from: date | None = None
    valid_until: date | None = None

    @model_validator(mode="after")
    def validate_interval(self):
        if self.valid_from and self.valid_until and self.valid_until < self.valid_from:
            raise ValueError("Assertion validity ends before it begins")
        for field in ("jurisdiction", "scope"):
            value = getattr(self, field)
            if value is not None and not value.strip():
                raise ValueError(f"Assertion {field} cannot be blank")
        return self


class AssertionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^assertion:[0-9a-f]{64}$")
    subject_id: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_value: str = Field(min_length=1)
    context: AssertionContext = Field(default_factory=AssertionContext)


class AssertionClassificationProposal(BaseModel):
    """A sourced classification proposal; it never represents adjudicated truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_assertion_id: str = Field(pattern=r"^assertion:[0-9a-f]{64}$")
    target_assertion_id: str = Field(pattern=r"^assertion:[0-9a-f]{64}$")
    classification: AssertionClassification
    classifier: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    source_sha256s: tuple[str, ...] = Field(min_length=1)
    ingestion_run_ids: tuple[str, ...] = Field(min_length=1)
    evidence_references: tuple[str, ...] = Field(min_length=1)
    adjudication_status: Literal["NOT_ADJUDICATED"] = "NOT_ADJUDICATED"

    @model_validator(mode="after")
    def validate_endpoints(self):
        if self.source_assertion_id == self.target_assertion_id:
            raise ValueError("An assertion cannot be classified against itself")
        if any(not value.strip() for value in (*self.source_sha256s, *self.ingestion_run_ids,
                                                *self.evidence_references, self.classifier,
                                                self.rationale)):
            raise ValueError("Classification provenance and rationale must be non-empty")
        if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
               for value in self.source_sha256s):
            raise ValueError("Classification source hashes must be lowercase SHA-256 values")
        for encoded in self.evidence_references:
            try:
                reference = json.loads(encoded)
            except (TypeError, ValueError) as exc:
                raise ValueError("Classification evidence references must be JSON objects") from exc
            if not isinstance(reference, dict):
                raise ValueError("Classification evidence references must be JSON objects")
            if (reference.get("source_sha256") not in self.source_sha256s
                    or reference.get("ingestion_run_id") not in self.ingestion_run_ids
                    or not isinstance(reference.get("quote"), str)
                    or not reference.get("quote")
                    or not isinstance(reference.get("start_offset"), int)
                    or not isinstance(reference.get("end_offset"), int)
                    or reference["end_offset"] - reference["start_offset"] != len(reference["quote"])):
                raise ValueError("Classification evidence is missing matching source provenance")
        return self


def assertion_identity(
    *, subject_id: str, predicate: str, object_value: str, context: AssertionContext
) -> str:
    canonical = json.dumps({
        "subject_id": subject_id,
        "predicate": predicate,
        "object_value": object_value,
        "context": context.model_dump(mode="json"),
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "assertion:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def classification_identity(proposal: AssertionClassificationProposal) -> str:
    canonical = json.dumps({
        "source": proposal.source_assertion_id,
        "target": proposal.target_assertion_id,
        "classification": proposal.classification.value,
        "classifier": proposal.classifier,
        "source_sha256s": sorted(proposal.source_sha256s),
        "ingestion_run_ids": sorted(proposal.ingestion_run_ids),
    }, sort_keys=True, separators=(",", ":"))
    return "assertion-classification:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def applicability_overlaps(existing: dict, incoming: AssertionContext) -> bool:
    """Only compare claims in the same jurisdiction/scope and overlapping periods."""
    for key in ("jurisdiction", "scope"):
        if existing.get(key) != getattr(incoming, key):
            return False

    def parse(value):
        if value is None:
            return None
        return value if isinstance(value, date) else date.fromisoformat(str(value))

    old_start, old_end = parse(existing.get("valid_from")), parse(existing.get("valid_until"))
    new_start, new_end = incoming.valid_from, incoming.valid_until
    if old_end is not None and new_start is not None and old_end < new_start:
        return False
    if new_end is not None and old_start is not None and new_end < old_start:
        return False
    return True
