"""End-to-end PDF ingestion orchestration and unrecognized-image crops."""

import base64
import math
from pathlib import Path

import pymupdf
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, model_validator
from typing import Literal, Self

from src.ingestion.chunking import ChunkingConfig, ChunkingResult, chunk_document
from src.ingestion.contracts import DocumentFeatures, DocumentRoutePlan
from src.ingestion.continuity import generate_continuity_candidates
from src.ingestion.continuity_contracts import ContinuityCandidate
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
from src.ingestion.preflight import inspect_pdf
from src.ingestion.router import route_document

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


class CPCRIngestionResponse(BaseModel):
    """API response: continuity results and preserved unrecognized images."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpcr: CPCRResult
    unrecognized_images: tuple[UnrecognizedImageCrop, ...]


async def classify_page_continuity(
    ingestion: PdfIngestionResult,
    client: ClassificationClient,
) -> CPCRResult:
    """Evaluate every adjacent-page candidate and classify only DEFER results."""
    document = ingestion.chunking.document
    chunks_by_index = {chunk.index: chunk for chunk in document.chunks}
    candidates = generate_continuity_candidates(document, "cpcr-candidate-2")
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
    source: Path, *, config: ChunkingConfig | None = None
) -> PdfIngestionResult:
    """Run preflight, routing, parsing, chunking and image preservation."""
    settings = config or ChunkingConfig()
    with _TRACER.start_as_current_span("ingestion.run") as root_span:
        try:
            with _TRACER.start_as_current_span("ingestion.preflight") as span:
                features = inspect_pdf(source)
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
                canonical = dispatch_document(source, route_plan)
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
