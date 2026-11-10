"""End-to-end PDF ingestion orchestration and unrecognized-image crops."""

import base64
import logging
import math
from pathlib import Path

import pymupdf
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal, Self

from src.ingestion.chunk_contracts import Chunk, ChunkSource
from src.ingestion.chunking import (
    ChunkingConfig,
    ChunkingResult,
    InvalidChunkingStateError,
    chunk_document,
)
from src.ingestion.contracts import DocumentFeatures, DocumentRoutePlan, NonNegativeInt
from src.ingestion.continuity import generate_continuity_candidates
from src.ingestion.continuity_contracts import (
    ContinuityCandidate,
    ContinuityRelation,
)
from src.ingestion.continuity_gateway import (
    CONTINUITY_GATEWAY_POLICY_VERSION,
    GatewayDecision,
    GatewayResult,
    evaluate_candidate,
)
from src.ingestion.continuity_graph import ClassificationClient, build_cpcr_graph
from src.ingestion.continuity_state import CPCRState
from src.ingestion.continuity_cdm_contracts import CDMClassificationResult
from src.ingestion.dispatcher import dispatch_document
from src.ingestion.parser_contracts import CanonicalDocument, SourceCoordinates
from src.ingestion.parsers import PageParser
from src.ingestion.preflight import inspect_pdf
from src.ingestion.router import route_document
from src.security.ingestion_screening.contracts import (
    AdmissionRecord,
    FileScreeningRecord,
    ScreeningFlag,
    ScreeningReport,
    ScreenedChunkedDocument,
)
from src.security.ingestion_screening.enums import AdmissionDecision

INGESTION_ROUTING_POLICY_VERSION = "ingestion-router-1.0.0"
MAX_CROP_PIXELS = 4_000_000
_TRACER = trace.get_tracer("graph_rag.ingestion")
_LOGGER = logging.getLogger(__name__)


class UnrecognizedImageCrop(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: int
    block_id: str
    reading_order: int
    source_coordinates: SourceCoordinates
    filename: str
    media_type: str = "image/png"
    data_base64: str
    ocr_status: Literal["TEXT_FOUND", "NO_TEXT", "FAILED"] | None = None
    screening_decision: AdmissionDecision | None = None
    screening_flags: tuple[ScreeningFlag, ...] = ()


class PdfIngestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    features: DocumentFeatures
    route_plan: DocumentRoutePlan
    canonical: CanonicalDocument
    chunking: ChunkingResult
    unrecognized_images: tuple[UnrecognizedImageCrop, ...]


class CPCRCandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: ContinuityCandidate
    gateway_result: GatewayResult
    classification_status: Literal[
        "NOT_REQUIRED", "CLASSIFIED", "REJECTED", "FAILED"
    ]
    cdm_result: CDMClassificationResult | None = None
    classification_error: str | None = None

    @model_validator(mode="after")
    def validate_classification_state(self) -> Self:
        if self.gateway_result.decision == GatewayDecision.DEFER:
            if self.classification_status == "NOT_REQUIRED":
                raise ValueError("Deferred candidates require a CDM outcome")
            if self.classification_status in {"CLASSIFIED", "REJECTED"} and (
                self.cdm_result is None
            ):
                raise ValueError("Classified candidates must contain a CDM result")
            if self.classification_status == "FAILED" and not self.classification_error:
                raise ValueError("Failed classifications require an error code")
            if self.classification_status == "REJECTED" and not self.classification_error:
                raise ValueError("Rejected classifications require a rejection code")
            if self.classification_status == "CLASSIFIED" and self.classification_error:
                raise ValueError("Classified candidates cannot contain an error code")
        elif (
            self.classification_status != "NOT_REQUIRED"
            or self.cdm_result is not None
            or self.classification_error is not None
        ):
            raise ValueError("Local gateway outcomes cannot contain CDM results")
        return self


class CPCRResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: Literal["COMPLETED", "PARTIAL"]
    candidates: tuple[CPCRCandidateResult, ...]

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        has_failures = any(
            candidate.classification_status == "FAILED"
            for candidate in self.candidates
        )
        if (self.outcome == "PARTIAL") != has_failures:
            raise ValueError("CPCR outcome must reflect classification failures")
        return self


class ChunkDisposition(BaseModel):
    """Compact admission status for each original BLP-1 chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_chunk_index: NonNegativeInt
    decision: AdmissionDecision
    reason_codes: tuple[str, ...]


class RevisedChunkAdmissionMapping(BaseModel):
    """Maps one revised chunk to the original chunks and their screening records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revised_chunk_index: NonNegativeInt
    original_chunk_indices: tuple[NonNegativeInt, ...] = Field(min_length=1)
    admission_records: tuple[AdmissionRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_original_records(self) -> Self:
        if self.original_chunk_indices != tuple(
            record.chunk_index for record in self.admission_records
        ):
            raise ValueError("Mapping indexes must match its original admission records")
        if tuple(sorted(set(self.original_chunk_indices))) != self.original_chunk_indices:
            raise ValueError("Original chunk indexes must be ordered and unique")
        return self


class RevisedChunkSet(BaseModel):
    """Original chunks with bounded source-backed context on continuations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    source_page_count: int
    policy_version: str
    context_policy: str
    chunks: tuple[Chunk, ...]
    contextualized_candidate_ids: tuple[str, ...]
    admission_records: tuple[AdmissionRecord, ...]
    admission_mappings: tuple[RevisedChunkAdmissionMapping, ...]
    screening_report: ScreeningReport

    @model_validator(mode="after")
    def validate_indices(self) -> Self:
        if tuple(chunk.index for chunk in self.chunks) != tuple(
            range(len(self.chunks))
        ):
            raise ValueError("Revised chunk indices must be contiguous from zero")
        if len(self.admission_mappings) != len(self.chunks):
            raise ValueError("Every revised chunk must have one admission mapping")
        original_records = {
            record.chunk_index: record for record in self.admission_records
        }
        for chunk, mapping in zip(self.chunks, self.admission_mappings, strict=True):
            if mapping.revised_chunk_index != chunk.index:
                raise ValueError("Admission mapping index must match revised chunk index")
            if any(
                original_records.get(record.chunk_index) != record
                for record in mapping.admission_records
            ):
                raise ValueError("Admission mapping records must come from original screening")
        return self


class CPCRIngestionResponse(BaseModel):
    """Public API response: final chunks and preserved image crops only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revised_chunks: tuple[Chunk, ...]
    unrecognized_images: tuple[UnrecognizedImageCrop, ...]
    chunk_dispositions: tuple[ChunkDisposition, ...]


async def classify_page_continuity(
    ingestion: PdfIngestionResult,
    client: ClassificationClient,
    screened: ScreenedChunkedDocument,
) -> CPCRResult:
    """Evaluate every adjacent-page candidate and classify only DEFER results."""
    document = ingestion.chunking.document
    if screened.document != document:
        raise ValueError("CPCR requires screening results for this exact chunked document")
    if (
        screened.report.file_screening is None
        or screened.report.file_screening.decision != AdmissionDecision.ADMITTED
    ):
        raise ValueError("CPCR requires an admitted G0 file-screening record")
    excluded_chunk_indices = frozenset(
        result.chunk_index
        for result in screened.report.chunk_results
        if result.decision
        not in {AdmissionDecision.ADMITTED, AdmissionDecision.ADMITTED_FLAGGED}
    )
    chunks_by_index = {chunk.index: chunk for chunk in document.chunks}
    candidates = generate_continuity_candidates(
        document,
        "cpcr-candidate-2",
        excluded_chunk_indices=excluded_chunk_indices,
    )
    graph = build_cpcr_graph(client, document)
    outcomes: list[CPCRCandidateResult] = []

    with _TRACER.start_as_current_span("ingestion.cpcr") as cpcr_span:
        cpcr_span.set_attribute("ingestion.cpcr.candidates", len(candidates))
        for candidate in candidates:
            gateway = evaluate_candidate(candidate, document)
            with _TRACER.start_as_current_span("ingestion.cpcr.candidate") as span:
                span.set_attribute("ingestion.cpcr.candidate_id", candidate.candidate_id)
                span.set_attribute("ingestion.cpcr.source_page", candidate.source.page_number)
                span.set_attribute("ingestion.cpcr.target_page", candidate.target.page_number)
                span.set_attribute("ingestion.cpcr.gateway_decision", gateway.decision.value)
                if gateway.decision != GatewayDecision.DEFER:
                    outcomes.append(
                        CPCRCandidateResult(
                            candidate=candidate,
                            gateway_result=GatewayResult(
                                decision=gateway.decision,
                                relation_type=gateway.relation_type,
                                evidence_codes=gateway.evidence_codes,
                                gateway_policy_version=CONTINUITY_GATEWAY_POLICY_VERSION,
                            ),
                            classification_status="NOT_REQUIRED",
                        )
                    )
                    continue

                source_chunk = chunks_by_index[candidate.source.chunk_index]
                target_chunk = chunks_by_index[candidate.target.chunk_index]
                state: CPCRState = {
                    "candidate": candidate,
                    "source_text": source_chunk.text,
                    "target_text": target_chunk.text,
                    "gateway_result": GatewayResult(
                        decision=gateway.decision,
                        relation_type=gateway.relation_type,
                        evidence_codes=gateway.evidence_codes,
                        gateway_policy_version=CONTINUITY_GATEWAY_POLICY_VERSION,
                    ),
                }
                try:
                    final_state = await graph.ainvoke(state)
                    cdm_result = final_state["cdm_result"]
                    rejection = final_state.get("classification_rejection")
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                    _LOGGER.warning(
                        "CPCR classification failed for candidate %s (error_type=%s)",
                        candidate.candidate_id,
                        type(exc).__name__,
                    )
                    outcomes.append(
                        CPCRCandidateResult(
                            candidate=candidate,
                            gateway_result=state["gateway_result"],
                            classification_status="FAILED",
                            classification_error=type(exc).__name__,
                        )
                    )
                else:
                    if rejection is not None:
                        span.set_attribute(
                            "ingestion.cpcr.classification_rejection", rejection
                        )
                        _LOGGER.warning(
                            "CPCR classification rejected for candidate %s (reason=%s)",
                            candidate.candidate_id,
                            rejection,
                        )
                        outcomes.append(
                            CPCRCandidateResult(
                                candidate=candidate,
                                gateway_result=state["gateway_result"],
                                classification_status="REJECTED",
                                cdm_result=cdm_result,
                                classification_error=rejection,
                            )
                        )
                        continue
                    span.set_attribute(
                        "ingestion.cpcr.relation", cdm_result.relation_type.value
                    )
                    if cdm_result.request_id:
                        span.set_attribute(
                            "ingestion.cpcr.request_id", cdm_result.request_id
                        )
                    outcomes.append(
                        CPCRCandidateResult(
                            candidate=candidate,
                            gateway_result=state["gateway_result"],
                            classification_status="CLASSIFIED",
                            cdm_result=cdm_result,
                        )
                    )

        has_failures = any(
            item.classification_status == "FAILED" for item in outcomes
        )
        cpcr_span.set_attribute("ingestion.cpcr.failed", sum(
            item.classification_status == "FAILED" for item in outcomes
        ))
    return CPCRResult(
        outcome="PARTIAL" if has_failures else "COMPLETED",
        candidates=tuple(outcomes),
    )


def revise_chunks_after_cpcr(
    ingestion: PdfIngestionResult,
    cpcr: CPCRResult,
    screened: ScreenedChunkedDocument,
    config: ChunkingConfig | None = None,
) -> RevisedChunkSet:
    """Add bounded prior-page text context to confirmed paragraph continuations.

    Original chunks remain separate and in source order. Context is duplicated
    only as a bounded prefix on the continuation chunk, with its exact source
    range retained in provenance.
    """
    original_document = ingestion.chunking.document
    if screened.document != original_document:
        raise ValueError("Screening results do not match the document being revised")
    settings = config or ChunkingConfig()
    admission_by_index = {
        result.chunk_index: result.decision for result in screened.report.chunk_results
    }
    admitted_indices = {
        chunk.index
        for chunk in original_document.chunks
        if admission_by_index.get(chunk.index)
        in {AdmissionDecision.ADMITTED, AdmissionDecision.ADMITTED_FLAGGED}
    }
    chunks_by_index = {
        chunk.index: chunk
        for chunk in original_document.chunks
        if chunk.index in admitted_indices
    }
    canonical_blocks = {
        block.block_id: block
        for page in ingestion.canonical.pages
        for block in page.blocks
    }
    context_by_target: dict[int, int] = {}
    candidate_by_target: dict[int, str] = {}
    for result in cpcr.candidates:
        if result.gateway_result.decision == GatewayDecision.ACCEPT:
            relation = result.gateway_result.relation_type
        elif result.classification_status == "CLASSIFIED" and result.cdm_result:
            relation = result.cdm_result.relation_type
        else:
            continue

        if relation not in {
            ContinuityRelation.SAME_PARAGRAPH,
        }:
            continue
        source_index = result.candidate.source.chunk_index
        target_index = result.candidate.target.chunk_index
        if source_index not in chunks_by_index or target_index not in chunks_by_index:
            continue
        if target_index in context_by_target:
            raise InvalidChunkingStateError(
                "Multiple positive continuity candidates target the same chunk"
            )
        context_by_target[target_index] = source_index
        candidate_by_target[target_index] = result.candidate.candidate_id

    from src.ingestion.segmentation import (
        _grapheme_boundaries,
        line_boundaries,
        word_boundaries,
    )

    chunk_limit = settings.max_chunk_chars
    overlap_limit = min(settings.overlap_chars, chunk_limit // 4)

    def context_prefix(
        source_index: int, target: Chunk
    ) -> tuple[str, ChunkSource] | None:
        source_chunk = chunks_by_index[source_index]
        if not source_chunk.sources or not target.sources:
            return None
        source_ref = source_chunk.sources[-1]
        if (
            source_ref.start_offset is None
            or source_ref.end_offset is None
            or source_ref.block_type not in {"TEXT", "OTHER"}
        ):
            return None
        source_block = canonical_blocks.get(source_ref.block_id)
        if source_block is None:
            return None
        source_slice = source_block.text[
            source_ref.start_offset : source_ref.end_offset
        ]
        if not source_slice or not source_chunk.text.endswith(source_slice):
            return None

        separator = "" if source_slice[-1:].isspace() or target.text[:1].isspace() else " "
        if overlap_limit <= 0:
            return None
        desired_start = max(
            source_ref.start_offset,
            source_ref.end_offset - overlap_limit,
        )
        line_starts = [
            boundary
            for boundary in line_boundaries(source_block.text)
            if desired_start <= boundary < source_ref.end_offset
        ]
        if line_starts:
            desired_start = max(line_starts)
        else:
            word_starts = [
                boundary
                for boundary in word_boundaries(source_block.text)
                if desired_start <= boundary < source_ref.end_offset
                and source_ref.end_offset - boundary <= overlap_limit
            ]
            if word_starts:
                desired_start = min(word_starts)
        grapheme_boundaries = _grapheme_boundaries(source_block.text)
        if desired_start not in set(grapheme_boundaries):
            safe_starts = [
                boundary
                for boundary in grapheme_boundaries
                if desired_start < boundary < source_ref.end_offset
            ]
            if not safe_starts:
                return None
            desired_start = min(safe_starts)
        overlap_text = source_block.text[desired_start : source_ref.end_offset]
        if not overlap_text:
            return None
        overlap_source = source_ref.model_copy(
            update={"start_offset": desired_start}
        )
        return overlap_text + separator, overlap_source

    def source_map_for_chunk(
        chunk: Chunk,
    ) -> tuple[tuple[ChunkSource, int] | None, ...] | None:
        rendered: list[str] = []
        offsets: list[tuple[ChunkSource, int] | None] = []
        previous_block_id: str | None = None
        for source in chunk.sources:
            if source.start_offset is None or source.end_offset is None:
                return None
            block = canonical_blocks.get(source.block_id)
            if block is None:
                return None
            if previous_block_id is not None and previous_block_id != source.block_id:
                rendered.append(settings.synthetic_separator)
                offsets.extend([None] * len(settings.synthetic_separator))
            part = block.text[source.start_offset : source.end_offset]
            rendered.append(part)
            offsets.extend(
                (source, offset)
                for offset in range(source.start_offset, source.end_offset)
            )
            previous_block_id = source.block_id
        if "".join(rendered) != chunk.text:
            return None
        return tuple(offsets)

    def sources_for_slice(
        source_map: tuple[tuple[ChunkSource, int] | None, ...],
        start: int,
        end: int,
    ) -> tuple[ChunkSource, ...]:
        sliced: list[ChunkSource] = []
        current_source: ChunkSource | None = None
        current_start = current_end = 0
        for entry in source_map[start:end]:
            if entry is None:
                if current_source is not None:
                    sliced.append(
                        current_source.model_copy(
                            update={
                                "start_offset": current_start,
                                "end_offset": current_end,
                            }
                        )
                    )
                    current_source = None
                continue
            source, offset = entry
            if (
                current_source is not None
                and current_source.block_id == source.block_id
                and current_end == offset
            ):
                current_end = offset + 1
                continue
            if current_source is not None:
                sliced.append(
                    current_source.model_copy(
                        update={
                            "start_offset": current_start,
                            "end_offset": current_end,
                        }
                    )
                )
            current_source = source
            current_start = offset
            current_end = offset + 1
        if current_source is not None:
            sliced.append(
                current_source.model_copy(
                    update={
                        "start_offset": current_start,
                        "end_offset": current_end,
                    }
                )
            )
        return tuple(sliced)

    revised_chunks: list[Chunk] = []
    admission_by_index = {
        record.chunk_index: record for record in screened.admission_records
    }
    admission_mappings: list[RevisedChunkAdmissionMapping] = []
    contextualized_candidate_ids: list[str] = []
    original_indexes = sorted(chunks_by_index)
    for original_index in original_indexes:
        original = chunks_by_index[original_index]
        if len(original.text) > chunk_limit:
            raise InvalidChunkingStateError(
                f"Original chunk {original.index} exceeds configured size limit"
            )
        source_index = context_by_target.get(original_index)
        prefix_and_source = (
            context_prefix(source_index, original)
            if source_index is not None
            else None
        )
        if prefix_and_source is None:
            revised_chunks.append(
                Chunk(index=len(revised_chunks), text=original.text, sources=original.sources)
            )
            mapped_indexes = (original_index,)
            admission_mappings.append(
                RevisedChunkAdmissionMapping(
                    revised_chunk_index=revised_chunks[-1].index,
                    original_chunk_indices=mapped_indexes,
                    admission_records=(admission_by_index[original_index],),
                )
            )
            continue

        if source_index is None:
            raise InvalidChunkingStateError(
                "Context prefix has no source chunk index"
            )
        prefix, overlap_source = prefix_and_source
        target_budget = chunk_limit - len(prefix)
        if target_budget < 1:
            revised_chunks.append(
                Chunk(index=len(revised_chunks), text=original.text, sources=original.sources)
            )
            admission_mappings.append(
                RevisedChunkAdmissionMapping(
                    revised_chunk_index=revised_chunks[-1].index,
                    original_chunk_indices=(original_index,),
                    admission_records=(admission_by_index[original_index],),
                )
            )
            continue

        source_map = source_map_for_chunk(original)
        if source_map is None:
            revised_chunks.append(
                Chunk(index=len(revised_chunks), text=original.text, sources=original.sources)
            )
            admission_mappings.append(
                RevisedChunkAdmissionMapping(
                    revised_chunk_index=revised_chunks[-1].index,
                    original_chunk_indices=(original_index,),
                    admission_records=(admission_by_index[original_index],),
                )
            )
            continue

        split_at = len(original.text)
        if len(original.text) > target_budget:
            from src.ingestion.segmentation import grapheme_safe_hard_cut

            word_cuts = [cut for cut in word_boundaries(original.text) if cut <= target_budget]
            split_at = (
                max(word_cuts)
                if word_cuts
                else grapheme_safe_hard_cut(
                    original.text, 0, target_budget, config=settings
                ).position
            )
        target_sources = sources_for_slice(source_map, 0, split_at)
        context_sources = (overlap_source, *target_sources)
        context_chunk = Chunk(
            index=len(revised_chunks),
            text=prefix + original.text[:split_at],
            sources=context_sources,
        )
        if len(context_chunk.text) > chunk_limit:
            raise InvalidChunkingStateError("Context chunk exceeds configured size limit")
        revised_chunks.append(context_chunk)
        context_indexes = tuple(sorted({source_index, original_index}))
        admission_mappings.append(
            RevisedChunkAdmissionMapping(
                revised_chunk_index=context_chunk.index,
                original_chunk_indices=context_indexes,
                admission_records=tuple(admission_by_index[index] for index in context_indexes),
            )
        )
        contextualized_candidate_ids.append(candidate_by_target[original_index])

        if split_at < len(original.text):
            remainder_sources = sources_for_slice(source_map, split_at, len(original.text))
            remainder = Chunk(
                index=len(revised_chunks),
                text=original.text[split_at:],
                sources=remainder_sources,
            )
            if len(remainder.text) > chunk_limit:
                raise InvalidChunkingStateError("Remainder chunk exceeds configured size limit")
            revised_chunks.append(remainder)
            admission_mappings.append(
                RevisedChunkAdmissionMapping(
                    revised_chunk_index=remainder.index,
                    original_chunk_indices=(original_index,),
                    admission_records=(admission_by_index[original_index],),
                )
            )

    return RevisedChunkSet(
        document_id=original_document.document_id,
        source_page_count=original_document.source_page_count,
        policy_version=original_document.policy_version,
        context_policy=(
            "SAME_PARAGRAPH: retain original chunks and prefix the target with "
            "up to overlap_chars of the source block's final line(s); "
            "duplicated context retains its source range; all chunks remain "
            "within max_chunk_chars"
        ),
        chunks=tuple(revised_chunks),
        contextualized_candidate_ids=tuple(contextualized_candidate_ids),
        admission_records=screened.admission_records,
        admission_mappings=tuple(admission_mappings),
        screening_report=screened.report,
    )


def _set_error(span, error: Exception) -> None:
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))


def crop_unrecognized_images(
    source: Path, canonical: CanonicalDocument
) -> tuple[UnrecognizedImageCrop, ...]:
    """Crop blank-text IMAGE blocks from the original PDF with provenance."""
    crops: list[UnrecognizedImageCrop] = []
    with pymupdf.open(source) as pdf:
        for page_result in canonical.pages:
            page = pdf.load_page(page_result.page_number - 1)
            for block in page_result.blocks:
                if block.block_type != "IMAGE" or block.text.strip():
                    continue
                coordinates = block.source_coordinates
                if coordinates.unit != "PDF_POINT":
                    raise ValueError(
                        "Image crop coordinates must use PDF_POINT units"
                    )
                if coordinates.origin == "BOTTOM_LEFT":
                    top = page.rect.height - coordinates.y1
                    bottom = page.rect.height - coordinates.y0
                else:
                    top, bottom = coordinates.y0, coordinates.y1
                clip = pymupdf.Rect(
                    coordinates.x0, top, coordinates.x1, bottom
                ) & page.rect
                if clip.is_empty or clip.width <= 0 or clip.height <= 0:
                    raise ValueError(
                        f"Image block {block.block_id} has no visible page area"
                    )
                scale = min(
                    2.0,
                    math.sqrt(MAX_CROP_PIXELS / (clip.width * clip.height)),
                )
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False
                )
                crop_bytes = pixmap.tobytes("png")
                crops.append(
                    UnrecognizedImageCrop(
                        page_number=block.page_number,
                        block_id=block.block_id,
                        reading_order=block.reading_order,
                        source_coordinates=coordinates,
                        filename=f"page-{block.page_number}-{block.block_id}.png",
                        data_base64=base64.b64encode(crop_bytes).decode("ascii"),
                    )
                )
    return tuple(crops)


def ingest_pdf(
    source: Path,
    *,
    file_screening: FileScreeningRecord,
    config: ChunkingConfig | None = None,
    fast_text_parser: PageParser | None = None,
    structured_parser: PageParser | None = None,
    layout_ocr_parser: PageParser | None = None,
    ocr_parser: PageParser | None = None,
    preflight_features: DocumentFeatures | None = None,
    prepared_route_plan: DocumentRoutePlan | None = None,
    parsed_canonical_document: CanonicalDocument | None = None,
    prepared_image_crops: tuple[UnrecognizedImageCrop, ...] | None = None,
) -> PdfIngestionResult:
    """Run preflight, routing, parsing, chunking and image preservation."""
    if (
        file_screening.decision != AdmissionDecision.ADMITTED
        or file_screening.source_sha256 is None
    ):
        raise ValueError("PDF parsing requires an admitted G0 file-screening record")
    prepared_values = (
        preflight_features,
        prepared_route_plan,
        parsed_canonical_document,
    )
    if any(value is not None for value in prepared_values) and any(
        value is None for value in prepared_values
    ):
        raise ValueError(
            "Precomputed features, route plan and parsed canonical document "
            "must be supplied together"
        )
    settings = config or ChunkingConfig()
    with _TRACER.start_as_current_span("ingestion.run") as root_span:
        try:
            with _TRACER.start_as_current_span("ingestion.preflight") as span:
                features = (
                    preflight_features
                    if preflight_features is not None
                    else inspect_pdf(source)
                )
                if file_screening.source_sha256 != features.content_sha256:
                    raise ValueError("G0 and PDF preflight source digests differ")
                if file_screening.page_count != features.page_count:
                    raise ValueError("G0 and PDF preflight page counts differ")
                span.set_attribute("ingestion.pdf.size_bytes", features.file_size_bytes)
                span.set_attribute("ingestion.pdf.page_count", features.page_count)
                span.set_attribute("ingestion.pdf.sha256", features.content_sha256)

            with _TRACER.start_as_current_span("ingestion.routing") as span:
                route_plan = (
                    prepared_route_plan
                    if prepared_route_plan is not None
                    else route_document(features, INGESTION_ROUTING_POLICY_VERSION)
                )
                if route_plan.page_count != features.page_count:
                    raise ValueError("Route plan page count differs from preflight")
                span.set_attribute(
                    "ingestion.routes",
                    ",".join(decision.route for decision in route_plan.decisions),
                )

            with _TRACER.start_as_current_span("ingestion.parsing") as span:
                canonical = (
                    parsed_canonical_document
                    if parsed_canonical_document is not None
                    else dispatch_document(
                        source,
                        route_plan,
                        fast_text_parser=fast_text_parser,
                        structured_parser=structured_parser,
                        layout_ocr_parser=layout_ocr_parser,
                        ocr_parser=ocr_parser,
                    )
                )
                if canonical.page_count != features.page_count:
                    raise ValueError("Canonical page count differs from preflight")
                span.set_attribute(
                    "ingestion.parsing.failed_pages",
                    sum(page.status == "FAILED" for page in canonical.pages),
                )

            with _TRACER.start_as_current_span("ingestion.chunking") as span:
                chunking = chunk_document(
                    canonical, features.content_sha256, settings
                )
                span.set_attribute(
                    "ingestion.chunks.count", len(chunking.document.chunks)
                )
                span.set_attribute(
                    "ingestion.chunking.outcome", chunking.report.outcome
                )

            with _TRACER.start_as_current_span(
                "ingestion.image_crops"
            ) as span:
                unrecognized_images = (
                    prepared_image_crops
                    if prepared_image_crops is not None
                    else crop_unrecognized_images(source, canonical)
                )
                span.set_attribute(
                    "ingestion.image_crops.count", len(unrecognized_images)
                )

            root_span.set_attribute("ingestion.document.id", features.content_sha256)
            return PdfIngestionResult(
                document_id=features.content_sha256,
                features=features,
                route_plan=route_plan,
                canonical=canonical,
                chunking=chunking,
                unrecognized_images=unrecognized_images,
            )
        except Exception as exc:
            _set_error(root_span, exc)
            raise
