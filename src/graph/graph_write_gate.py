"""Pure validation for extracted triples before any Neo4j write."""

from __future__ import annotations

from typing import Annotated, Self
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.domain.graph_schema import validate_node
from src.domain.ontology import EpistemicMode, RelationType, validate_relation
from src.ingestion.chunk_contracts import Chunk, ChunkedDocument
from src.ingestion.contracts import NonEmptyStr, NonNegativeInt, PositiveInt
from src.security.ingestion_screening.contracts import ScreenedChunkedDocument
from src.security.ingestion_screening.enums import AdmissionDecision


class EvidenceSpan(BaseModel):
    """Exact quote range into one admitted, original BLP-1 chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_index: NonNegativeInt
    start_offset: NonNegativeInt
    end_offset: PositiveInt
    quote: Annotated[str, StringConstraints(min_length=1)]

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.end_offset <= self.start_offset:
            raise ValueError("Evidence span must be a non-empty half-open range")
        return self


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_type: NonEmptyStr
    properties: dict[str, object]
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    epistemic_mode: EpistemicMode = EpistemicMode.VERBATIM_EXTRACTION
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_properties(self) -> Self:
        validate_node(self.node_type, self.properties)
        return self


class GraphRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: NonEmptyStr | None = None
    source_id: NonEmptyStr
    relation_type: NonEmptyStr
    target_id: NonEmptyStr
    evidence_spans: tuple[EvidenceSpan, ...] = Field(min_length=1)
    epistemic_mode: EpistemicMode = EpistemicMode.VERBATIM_EXTRACTION
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def assign_stable_id(self) -> Self:
        if self.id is None:
            identity = json.dumps(
                [self.source_id, self.relation_type.upper(), self.target_id],
                separators=(",", ":"),
            )
            object.__setattr__(
                self, "id", "relation:" + hashlib.sha256(identity.encode()).hexdigest()
            )
        return self


class GraphWriteBatch(BaseModel):
    """All extracted nodes and relations to be checked as one write unit."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    document: ChunkedDocument
    screened: ScreenedChunkedDocument
    ingestion_run_id: NonEmptyStr | None = None
    nodes: tuple[GraphNode, ...]
    relationships: tuple[GraphRelationship, ...]

    @model_validator(mode="after")
    def validate_before_write(self) -> Self:
        _validate_batch_fields(self)
        return self


def _validate_batch_fields(batch: GraphWriteBatch) -> None:
    if batch.screened.document != batch.document:
        raise ValueError("Screening results do not match the source document")
    admitted = {
        result.chunk_index
        for result in batch.screened.report.chunk_results
        if result.decision
        in {AdmissionDecision.ADMITTED, AdmissionDecision.ADMITTED_FLAGGED}
    }
    chunks = {chunk.index: chunk for chunk in batch.document.chunks}
    node_types = {}
    for node in batch.nodes:
        validate_node(node.node_type, node.properties)
        if node.epistemic_mode is not EpistemicMode.VERBATIM_EXTRACTION:
            raise ValueError("Inferred or rule-derived nodes are not admitted as graph facts")
        if node.confidence < 0.75:
            raise ValueError("Graph entity confidence is below the admission threshold")
        node_id = node.properties.get("id")
        if not isinstance(node_id, str) or not node_id.strip() or node_id in node_types:
            raise ValueError("Graph node IDs must be present and unique")
        node_type = node.node_type.upper()
        if node_type in {"DOCUMENT", "CHUNK"}:
            raise ValueError("Structural DOCUMENT/CHUNK nodes are managed by structural ingestion")
        node_types[node_id] = node_type
        if node_type not in {"DOCUMENT", "CHUNK"} and not node.evidence_spans:
            raise ValueError("Extracted graph entities require source evidence")
        for evidence in node.evidence_spans:
            _validate_evidence(evidence, chunks, admitted)
    for edge in batch.relationships:
        if edge.relation_type.upper() in {"HAS_CHUNK", "NEXT_CHUNK", "EXTRACTED_FROM"}:
            raise ValueError("Structural provenance relationships are managed by structural ingestion")
        if edge.epistemic_mode is not EpistemicMode.VERBATIM_EXTRACTION:
            raise ValueError("Inferred or rule-derived relationships are not admitted as graph facts")
        if edge.confidence < 0.75:
            raise ValueError("Graph relationship confidence is below the admission threshold")
        try:
            source_type = node_types[edge.source_id]
            target_type = node_types[edge.target_id]
        except KeyError as exc:
            raise ValueError("Relationship endpoints must be in the write batch") from exc
        if not validate_relation(source_type, edge.relation_type, target_type):
            raise ValueError("Relationship type or endpoint types violate the ontology")
        for evidence in edge.evidence_spans:
            _validate_evidence(evidence, chunks, admitted)


def _validate_evidence(
    evidence: EvidenceSpan, chunks: dict[int, Chunk], admitted: set[int]
) -> None:
    if evidence.chunk_index not in admitted:
        raise ValueError("Evidence may only cite admitted chunks")
    chunk = chunks.get(evidence.chunk_index)
    if chunk is None or evidence.end_offset > len(chunk.text):
        raise ValueError("Evidence span exceeds its source chunk")
    if chunk.text[evidence.start_offset : evidence.end_offset] != evidence.quote:
        raise ValueError("Evidence quote does not match its source span")
    if not chunk.sources or len({source.page_number for source in chunk.sources}) != 1:
        raise ValueError("Evidence chunk must have page-local source provenance")


def validate_graph_write_batch(batch: GraphWriteBatch) -> None:
    """Validate a batch before a caller executes any graph mutations."""
    if not isinstance(batch, GraphWriteBatch):
        raise TypeError("Expected a validated GraphWriteBatch")
    # Pydantic's frozen model is shallow; revalidate mutable property mappings.
    _validate_batch_fields(batch)
