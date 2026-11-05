"""Immutable sidecar models for ingestion screening and admission."""

from __future__ import annotations

import hashlib
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.ingestion.chunk_contracts import Chunk, ChunkedDocument
from src.ingestion.contracts import NonEmptyStr, NonNegativeInt, PositiveInt
from src.security.ingestion_screening.enums import (
    AdmissionDecision,
    DetectionView,
    FlagSeverity,
    ScreeningStage,
    TrustTier,
)

_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_NonEmptyTuple = Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]


class DocumentContext(BaseModel):
    """Provenance supplied by the caller or existing ingestion contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: NonEmptyStr | None = None
    tenant_id: NonEmptyStr | None = None
    uploader_id: NonEmptyStr | None = None
    ingestion_run_id: NonEmptyStr | None = None
    source_sha256: _Sha256
    canonical_document_digest: _Sha256
    chunking_policy_version: NonEmptyStr
    source_page_count: PositiveInt
    trust_tier: TrustTier

    @property
    def has_required_provenance(self) -> bool:
        return all(
            (
                self.source_id,
                self.tenant_id,
                self.uploader_id,
                self.ingestion_run_id,
            )
        )


class ScreeningFlag(BaseModel):
    """A detector finding mapped to exact offsets in an original chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: NonEmptyStr
    stage: ScreeningStage
    severity: FlagSeverity
    detection_view: DetectionView
    start_offset: NonNegativeInt | None = None
    end_offset: NonNegativeInt | None = None
    related_chunk_index: NonNegativeInt | None = None
    matched_value: str | None = None
    urls: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("Both flag offsets must be present or both absent")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset <= self.start_offset
        ):
            raise ValueError("Flag range must be a non-empty half-open range")
        if self.detection_view == DetectionView.BOUNDARY and (
            self.related_chunk_index is None
        ):
            raise ValueError("Boundary flags must identify the related chunk")
        return self


class ChunkScreeningResult(BaseModel):
    """Screening verdict and immutable identity for one original chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_index: NonNegativeInt
    chunk_digest: _Sha256
    chunk_text_length: NonNegativeInt
    decision: AdmissionDecision
    flags: tuple[ScreeningFlag, ...] = ()
    justification_codes: _NonEmptyTuple
    screening_policy_version: NonEmptyStr
    screening_policy_digest: _Sha256
    source_id: NonEmptyStr | None = None
    tenant_id: NonEmptyStr | None = None
    uploader_id: NonEmptyStr | None = None
    ingestion_run_id: NonEmptyStr | None = None
    source_sha256: _Sha256
    canonical_document_digest: _Sha256
    chunking_policy_version: NonEmptyStr
    trust_tier: TrustTier

    @model_validator(mode="after")
    def validate_flag_offsets(self) -> Self:
        for flag in self.flags:
            if (
                flag.end_offset is not None
                and flag.end_offset > self.chunk_text_length
            ):
                raise ValueError("Flag range exceeds the original chunk text")
        return self


class AdmissionRecord(BaseModel):
    """Downstream provenance and admission metadata for an unchanged chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_index: NonNegativeInt
    chunk_digest: _Sha256
    decision: AdmissionDecision
    source_id: NonEmptyStr | None = None
    tenant_id: NonEmptyStr | None = None
    uploader_id: NonEmptyStr | None = None
    ingestion_run_id: NonEmptyStr | None = None
    source_sha256: _Sha256
    canonical_document_digest: _Sha256
    chunking_policy_version: NonEmptyStr
    screening_policy_version: NonEmptyStr
    screening_policy_digest: _Sha256
    trust_tier: TrustTier
    justification_codes: _NonEmptyTuple


class FileScreeningRecord(BaseModel):
    """G0 file-level verdict, kept separate from chunk-level results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_sha256: _Sha256 | None = None
    file_size_bytes: NonNegativeInt
    page_count: NonNegativeInt | None = None
    decision: AdmissionDecision
    flags: tuple[ScreeningFlag, ...] = ()
    scanner: NonEmptyStr | None = None
    scanner_version: NonEmptyStr | None = None
    scanner_database_version: NonEmptyStr | None = None
    screening_policy_version: NonEmptyStr
    screening_policy_digest: _Sha256
    justification_codes: _NonEmptyTuple

    @model_validator(mode="after")
    def validate_admitted_record(self) -> Self:
        if self.decision == AdmissionDecision.ADMITTED and (
            self.source_sha256 is None or self.scanner is None
        ):
            raise ValueError("An admitted file requires a digest and scanner verdict")
        return self


class ScreeningReport(BaseModel):
    """One deterministic batch report with exactly one result per chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: NonEmptyStr
    input_chunk_count: NonNegativeInt
    chunk_results: tuple[ChunkScreeningResult, ...]
    screening_policy_version: NonEmptyStr
    screening_policy_digest: _Sha256
    quarantined_document: bool = False
    file_screening: FileScreeningRecord | None = None

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        if len(self.chunk_results) != self.input_chunk_count:
            raise ValueError("Screening report must contain one result per input chunk")
        indices = tuple(result.chunk_index for result in self.chunk_results)
        if indices != tuple(range(self.input_chunk_count)):
            raise ValueError("Screening results must preserve original chunk indexes")
        for result in self.chunk_results:
            if (
                result.screening_policy_version != self.screening_policy_version
                or result.screening_policy_digest != self.screening_policy_digest
            ):
                raise ValueError("Chunk results must use the report screening policy")
            if self.file_screening is not None and (
                self.file_screening.decision != AdmissionDecision.ADMITTED
            ):
                raise ValueError("A blocked G0 file cannot have a screening report")
            if self.file_screening is not None and (
                result.source_sha256 != self.file_screening.source_sha256
            ):
                raise ValueError("G0 and chunk results must identify the same source")
        if self.file_screening is not None and (
            self.file_screening.screening_policy_version != self.screening_policy_version
            or self.file_screening.screening_policy_digest != self.screening_policy_digest
        ):
            raise ValueError("File screening must use the report screening policy")
        if self.quarantined_document and any(
            result.decision not in {
                AdmissionDecision.QUARANTINED,
                AdmissionDecision.REJECTED,
            }
            for result in self.chunk_results
        ):
            raise ValueError("A quarantined document cannot retain admitted chunks")
        return self


class ScreenedChunkedDocument(BaseModel):
    """Sidecar admission metadata paired with, but never mutating, source chunks."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    document: ChunkedDocument
    admission_records: tuple[AdmissionRecord, ...]
    report: ScreeningReport

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        chunks: tuple[Chunk, ...] = self.document.chunks
        records = self.admission_records
        if len(chunks) != len(records) or len(chunks) != len(self.report.chunk_results):
            raise ValueError("Every original chunk must have exactly one admission record")
        for chunk, record, result in zip(
            chunks, records, self.report.chunk_results, strict=True
        ):
            if chunk.index != record.chunk_index or chunk.index != result.chunk_index:
                raise ValueError("Admission metadata must preserve original chunk IDs")
            if record.chunk_digest != result.chunk_digest:
                raise ValueError("Admission and screening chunk digests must match")
            for field_name in (
                "decision",
                "source_id",
                "tenant_id",
                "uploader_id",
                "ingestion_run_id",
                "source_sha256",
                "canonical_document_digest",
                "chunking_policy_version",
                "screening_policy_version",
                "screening_policy_digest",
                "trust_tier",
                "justification_codes",
            ):
                if getattr(record, field_name) != getattr(result, field_name):
                    raise ValueError(
                        f"Admission record and screening result disagree on {field_name}"
                    )
            if result.chunk_digest != hashlib.sha256(
                chunk.text.encode("utf-8")
            ).hexdigest():
                raise ValueError("Screening metadata does not match original chunk text")
        if self.report.document_id != self.document.document_id:
            raise ValueError("Screening report document ID does not match chunks")
        return self
