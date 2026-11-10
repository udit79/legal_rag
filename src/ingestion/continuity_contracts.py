"""Provider-neutral contracts for adjacent-page chunk continuity."""

from enum import Enum
from hashlib import sha256
import json
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ingestion.chunk_contracts import Chunk, ChunkedDocument
from src.ingestion.contracts import NonEmptyStr, NonNegativeInt, PositiveInt


class ContinuityRelation(str, Enum):
    SAME_PARAGRAPH = "SAME_PARAGRAPH"
    TABLE_CONTINUATION = "TABLE_CONTINUATION"
    NO_CONTINUATION = "NO_CONTINUATION"
    UNKNOWN = "UNKNOWN"


class AssessmentMethod(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    CDM_CLASSIFY = "CDM_CLASSIFY"
    MANUAL = "MANUAL"


class ChunkReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: NonEmptyStr
    chunk_index: NonNegativeInt
    page_number: PositiveInt


def deterministic_candidate_id(
    document_id: str,
    chunking_policy_version: str,
    source_chunk_index: int,
    target_chunk_index: int,
) -> str:
    """Hash an unambiguous, versioned serialization of candidate identity."""
    identity = json.dumps(
        [
            "cpcr-candidate-v1",
            document_id,
            chunking_policy_version,
            source_chunk_index,
            target_chunk_index,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256(identity.encode("utf-8")).hexdigest()


class ContinuityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: NonEmptyStr
    source: ChunkReference
    target: ChunkReference
    chunking_policy_version: NonEmptyStr
    candidate_policy_version: NonEmptyStr

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        if self.source.document_id != self.target.document_id:
            raise ValueError("Source and target must belong to the same document")
        if self.target.page_number != self.source.page_number + 1:
            raise ValueError("Target page must immediately follow source page")
        if self.target.chunk_index <= self.source.chunk_index:
            raise ValueError("Target chunk index must be greater than source index")
        expected_id = deterministic_candidate_id(
            self.source.document_id,
            self.chunking_policy_version,
            self.source.chunk_index,
            self.target.chunk_index,
        )
        if self.candidate_id != expected_id:
            raise ValueError("Candidate ID does not match its deterministic identity")
        return self


class ContinuityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: ContinuityCandidate
    relation_type: ContinuityRelation
    method: AssessmentMethod
    evidence_codes: tuple[NonEmptyStr, ...] = Field(min_length=1)
    assessment_policy_version: NonEmptyStr


def _chunk_for_reference(
    document: ChunkedDocument,
    reference: ChunkReference,
) -> Chunk:
    if reference.document_id != document.document_id:
        raise ValueError("Chunk reference document ID does not match document")
    if reference.page_number > document.source_page_count:
        raise ValueError("Referenced page number exceeds document page count")
    chunk = next(
        (item for item in document.chunks if item.index == reference.chunk_index),
        None,
    )
    if chunk is None:
        raise ValueError(f"Referenced chunk {reference.chunk_index} does not exist")
    pages = {source.page_number for source in chunk.sources}
    if len(pages) != 1:
        raise ValueError("Referenced chunk must have provenance from exactly one page")
    if next(iter(pages)) != reference.page_number:
        raise ValueError("Referenced page number does not match chunk provenance")
    return chunk


def _chunks_on_page(document: ChunkedDocument, page_number: int) -> list[Chunk]:
    page_chunks: list[Chunk] = []
    for chunk in document.chunks:
        if not chunk.sources:
            raise ValueError(f"Chunk {chunk.index} has no source provenance")
        pages = {source.page_number for source in chunk.sources}
        if len(pages) != 1:
            raise ValueError(f"Chunk {chunk.index} has cross-page provenance")
        if next(iter(pages)) == page_number:
            page_chunks.append(chunk)
    if not page_chunks:
        raise ValueError(f"No chunks found for page {page_number}")
    return page_chunks


def validate_candidate_against_document(
    candidate: ContinuityCandidate,
    document: ChunkedDocument,
) -> None:
    """Validate references, chunking version, and BLP page-local provenance."""
    if candidate.chunking_policy_version != document.policy_version:
        raise ValueError("Candidate was generated for a different chunking version")
    if (
        candidate.source.document_id != document.document_id
        or candidate.target.document_id != document.document_id
    ):
        raise ValueError("Candidate document ID does not match document")
    _chunk_for_reference(document, candidate.source)
    _chunk_for_reference(document, candidate.target)
    # Candidate selection is flow-aware: a page boundary may have independent
    # text and table candidates, so references need not be the page's global
    # final and first chunks. Page locality and adjacency remain mandatory.
    _chunks_on_page(document, candidate.source.page_number)
    _chunks_on_page(document, candidate.target.page_number)


def validate_assessment_against_document(
    assessment: ContinuityAssessment,
    document: ChunkedDocument,
) -> None:
    """Validate document-dependent assessment constraints and source types."""
    validate_candidate_against_document(assessment.candidate, document)
    if assessment.relation_type == ContinuityRelation.TABLE_CONTINUATION:
        allowed_types = {"TABLE"}
        error = "TABLE_CONTINUATION requires TABLE-only source chunks"
    elif assessment.relation_type == ContinuityRelation.SAME_PARAGRAPH:
        allowed_types = {"TEXT", "OTHER"}
        error = "SAME_PARAGRAPH requires TEXT or OTHER source chunks"
    else:
        return

    for reference in (assessment.candidate.source, assessment.candidate.target):
        chunk = _chunk_for_reference(document, reference)
        if not chunk.sources or any(
            source.block_type not in allowed_types for source in chunk.sources
        ):
            raise ValueError(error)
