"""Per-ingestion-run visual candidate processing with isolated failure status."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Literal

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.graph.corpus_reader import CorpusDocument
from src.ingestion.chunk_contracts import ChunkSource
from src.graph.visual_observation_writer import persist_visual_observations
from src.graph.visual_observations import (
    VisualGraphExtractor,
    extract_document_visual_observations,
)

_TRACER = trace.get_tracer("graph_rag.graph.visual_ingestion")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualGraphRunStatus:
    source_sha256: str
    ingestion_run_id: str
    status: Literal["SUCCEEDED", "NO_VISUAL_EVIDENCE", "FAILED"]
    observations: int = 0
    candidates: int = 0
    relationships: int = 0
    error_code: str | None = None
    error_type: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def has_admitted_visual_evidence(document: CorpusDocument) -> bool:
    return any(
        chunk.admission_decisions
        and all(decision in {"ADMITTED", "ADMITTED_FLAGGED"}
                for decision in chunk.admission_decisions)
        and any(ChunkSource.model_validate(source).extraction_method == "IMAGE_VLM"
                for source in chunk.sources)
        for chunk in document.chunks
    )


async def process_visual_graph_documents(
    db_connection,
    documents: tuple[CorpusDocument, ...],
    extractor: VisualGraphExtractor,
    asset_resolver,
) -> tuple[VisualGraphRunStatus, ...]:
    """Run after structural ingestion; each run has its own failure boundary and transaction."""
    statuses: list[VisualGraphRunStatus] = []
    for document in documents:
        with _TRACER.start_as_current_span("graph.visual.process_ingestion_run") as span:
            span.set_attribute("document.sha256", document.source_sha256)
            span.set_attribute("ingestion.run_id", document.ingestion_run_id)
            try:
                observations = await extract_document_visual_observations(
                    extractor, document, asset_resolver
                )
                if observations:
                    counts = await persist_visual_observations(db_connection, observations)
                    status = VisualGraphRunStatus(
                        document.source_sha256, document.ingestion_run_id, "SUCCEEDED",
                        observations=counts["observations"],
                        candidates=counts["candidates"],
                        relationships=counts["candidate_relationships"],
                    )
                else:
                    status = VisualGraphRunStatus(
                        document.source_sha256, document.ingestion_run_id,
                        "NO_VISUAL_EVIDENCE",
                    )
                span.set_attribute("visual_graph.status", status.status)
                span.set_attribute("visual_graph.observations", status.observations)
                span.set_attribute("visual_graph.candidates", status.candidates)
                span.set_attribute("visual_graph.relationships", status.relationships)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                span.set_attribute("visual_graph.status", "FAILED")
                span.set_attribute("visual_graph.error_code", "VISUAL_GRAPH_PROCESSING_FAILED")
                span.set_attribute("visual_graph.error_type", type(exc).__name__)
                _LOGGER.exception(
                    "Visual graph processing failed for ingestion run",
                    extra={
                        "source_sha256": document.source_sha256,
                        "ingestion_run_id": document.ingestion_run_id,
                        "visual_graph_status": "FAILED",
                        "visual_graph_error_type": type(exc).__name__,
                    },
                )
                status = VisualGraphRunStatus(
                    document.source_sha256, document.ingestion_run_id, "FAILED",
                    error_code="VISUAL_GRAPH_PROCESSING_FAILED",
                    error_type=type(exc).__name__,
                )
            statuses.append(status)
    return tuple(statuses)


def visual_graph_failure_statuses(
    documents: tuple[CorpusDocument, ...], error_code: str, exc: Exception
) -> tuple[VisualGraphRunStatus, ...]:
    """Record setup failures against each run without changing structural success."""
    statuses = []
    for document in documents:
        with _TRACER.start_as_current_span("graph.visual.setup_failed") as span:
            span.set_attribute("document.sha256", document.source_sha256)
            span.set_attribute("ingestion.run_id", document.ingestion_run_id)
            span.set_attribute("visual_graph.status", "FAILED")
            span.set_attribute("visual_graph.error_code", error_code)
            span.set_attribute("visual_graph.error_type", type(exc).__name__)
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
        _LOGGER.error(
            "Visual graph setup failed for ingestion run",
            extra={
                "source_sha256": document.source_sha256,
                "ingestion_run_id": document.ingestion_run_id,
                "visual_graph_status": "FAILED",
                "visual_graph_error_code": error_code,
                "visual_graph_error_type": type(exc).__name__,
            },
        )
        statuses.append(VisualGraphRunStatus(
            document.source_sha256, document.ingestion_run_id, "FAILED",
            error_code=error_code, error_type=type(exc).__name__,
        ))
    return tuple(statuses)
