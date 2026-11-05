"""End-to-end PDF ingestion orchestration and unrecognized-image crops."""

import base64
import math
from pathlib import Path

import pymupdf
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal, Self

from src.ingestion.chunk_contracts import Chunk
from src.ingestion.chunking import ChunkingConfig, ChunkingResult, chunk_document
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
    ScreeningReport,
    ScreenedChunkedDocument,
)
from src.security.ingestion_screening.enums import AdmissionDecision

INGESTION_ROUTING_POLICY_VERSION = "ingestion-router-1.0.0"
MAX_CROP_PIXELS = 4_000_000
_TRACER = trace.get_tracer("graph_rag.ingestion")


class UnrecognizedImageCrop(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: int
    block_id: str
    reading_order: int
    source_coordinates: SourceCoordinates
    filename: str
    media_type: str = "image/png"
    data_base64: str


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
    classification_status: Literal["NOT_REQUIRED", "CLASSIFIED", "FAILED"]
    cdm_result: CDMClassificationResult | None = None
    classification_error: str | None = None

    @model_validator(mode="after")
    def validate_classification_state(self) -> Self:
        if self.gateway_result.decision == GatewayDecision.DEFER:
            if self.classification_status == "NOT_REQUIRED":
                raise ValueError("Deferred candidates require a CDM outcome")
            if (self.classification_status == "CLASSIFIED") != (
                self.cdm_result is not None
            ):
                raise ValueError("Classified candidates must contain a CDM result")
            if (self.classification_status == "FAILED") != bool(
                self.classification_error
            ):
                raise ValueError("Failed classifications require an error code")
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
    """Chunks after applying positive CPCR page-continuity relationships."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    source_page_count: int
    policy_version: str
    merge_separator_policy: str
    chunks: tuple[Chunk, ...]
    merged_candidate_ids: tuple[str, ...]
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
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                    outcomes.append(
                        CPCRCandidateResult(
                            candidate=candidate,
                            gateway_result=state["gateway_result"],
                            classification_status="FAILED",
                            classification_error=type(exc).__name__,
                        )
                    )
                else:
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
) -> RevisedChunkSet:
    """Merge components connected by positive, validated CPCR outcomes."""
    original_document = ingestion.chunking.document
    if screened.document != original_document:
        raise ValueError("Screening results do not match the document being revised")
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
    parent = {index: index for index in chunks_by_index}
    merged_candidate_ids: list[str] = []
    merge_relations: dict[tuple[int, int], ContinuityRelation] = {}

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != index:
            next_index = parent[index]
            parent[index] = root
            index = next_index
        return root

    def union(source_index: int, target_index: int) -> None:
        source_root = find(source_index)
        target_root = find(target_index)
        if source_root != target_root:
            parent[target_root] = source_root

    for result in cpcr.candidates:
        if result.gateway_result.decision == GatewayDecision.ACCEPT:
            relation = result.gateway_result.relation_type
        elif result.classification_status == "CLASSIFIED" and result.cdm_result:
            relation = result.cdm_result.relation_type
        else:
            continue

        if relation not in {
            ContinuityRelation.SAME_PARAGRAPH,
            ContinuityRelation.TABLE_CONTINUATION,
        }:
            continue
        source_index = result.candidate.source.chunk_index
        target_index = result.candidate.target.chunk_index
        if source_index not in chunks_by_index or target_index not in chunks_by_index:
            continue
        union(source_index, target_index)
        merge_relations[(source_index, target_index)] = relation
        merged_candidate_ids.append(result.candidate.candidate_id)

    components: dict[int, list[int]] = {}
    for chunk_index in chunks_by_index:
        components.setdefault(find(chunk_index), []).append(chunk_index)

    ordered_components = sorted(components.values(), key=min)
    revised_chunks = []
    admission_by_index = {
        record.chunk_index: record for record in screened.admission_records
    }
    admission_mappings: list[RevisedChunkAdmissionMapping] = []
    for revised_index, component in enumerate(ordered_components):
        component_indices = sorted(component)
        originals = [chunks_by_index[index] for index in component_indices]
        text_parts = [originals[0].text]
        component_relation_values = [
            relation
            for (source, target), relation in merge_relations.items()
            if source in component and target in component
        ]
        default_relation = (
            ContinuityRelation.SAME_PARAGRAPH
            if ContinuityRelation.SAME_PARAGRAPH in component_relation_values
            else ContinuityRelation.TABLE_CONTINUATION
        )
        for left_index, right_index, original in zip(
            component_indices[:-1],
            component_indices[1:],
            originals[1:],
            strict=True,
        ):
            previous_text = text_parts[-1]
            relation = merge_relations.get(
                (left_index, right_index), default_relation
            )
            if relation == ContinuityRelation.TABLE_CONTINUATION:
                separator = (
                    ""
                    if previous_text.endswith(("\n", "\r"))
                    or original.text.startswith(("\n", "\r"))
                    else "\n"
                )
            else:
                separator = (
                    ""
                    if previous_text[-1:].isspace()
                    or original.text[:1].isspace()
                    else " "
                )
            text_parts.append(separator + original.text)
        text = "".join(text_parts)
        sources = tuple(source for chunk in originals for source in chunk.sources)
        revised_chunks.append(
            Chunk(index=revised_index, text=text, sources=sources)
        )
        original_records = tuple(
            admission_by_index[index] for index in component_indices
        )
        admission_mappings.append(
            RevisedChunkAdmissionMapping(
                revised_chunk_index=revised_index,
                original_chunk_indices=tuple(component_indices),
                admission_records=original_records,
            )
        )

    return RevisedChunkSet(
        document_id=original_document.document_id,
        source_page_count=original_document.source_page_count,
        policy_version=original_document.policy_version,
        merge_separator_policy=(
            "SAME_PARAGRAPH: insert one space only when needed; "
            "TABLE_CONTINUATION: insert one line break only when needed"
        ),
        chunks=tuple(revised_chunks),
        merged_candidate_ids=tuple(merged_candidate_ids),
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
) -> PdfIngestionResult:
    """Run preflight, routing, parsing, chunking and image preservation."""
    if (
        file_screening.decision != AdmissionDecision.ADMITTED
        or file_screening.source_sha256 is None
    ):
        raise ValueError("PDF parsing requires an admitted G0 file-screening record")
    settings = config or ChunkingConfig()
    with _TRACER.start_as_current_span("ingestion.run") as root_span:
        try:
            with _TRACER.start_as_current_span("ingestion.preflight") as span:
                features = inspect_pdf(source)
                if file_screening.source_sha256 != features.content_sha256:
                    raise ValueError("G0 and PDF preflight source digests differ")
                if file_screening.page_count != features.page_count:
                    raise ValueError("G0 and PDF preflight page counts differ")
                span.set_attribute("ingestion.pdf.size_bytes", features.file_size_bytes)
                span.set_attribute("ingestion.pdf.page_count", features.page_count)
                span.set_attribute("ingestion.pdf.sha256", features.content_sha256)

            with _TRACER.start_as_current_span("ingestion.routing") as span:
                route_plan = route_document(
                    features, INGESTION_ROUTING_POLICY_VERSION
                )
                span.set_attribute(
                    "ingestion.routes",
                    ",".join(decision.route for decision in route_plan.decisions),
                )

            with _TRACER.start_as_current_span("ingestion.parsing") as span:
                canonical = dispatch_document(
                    source,
                    route_plan,
                    fast_text_parser=fast_text_parser,
                    structured_parser=structured_parser,
                    layout_ocr_parser=layout_ocr_parser,
                    ocr_parser=ocr_parser,
                )
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
                unrecognized_images = crop_unrecognized_images(source, canonical)
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
