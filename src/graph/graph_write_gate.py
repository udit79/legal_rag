"""Pure validation for extracted triples before any Neo4j write."""

from __future__ import annotations

from typing import Annotated, Mapping, Self
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.domain.graph_schema import validate_node
from src.domain.ontology import EpistemicMode, NodeType, RelationType, validate_relation
from src.graph.assertion_contracts import AssertionContext
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


class EvidenceSupportAttestation(BaseModel):
    """Independent verifier result bound to one claim and exact source quote."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: NonEmptyStr
    model: NonEmptyStr
    claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    supporting_quote_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    supported: bool
    directly_stated: bool
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


VISUAL_FACT_EVIDENCE_POLICY = "DENY_IMAGE_VLM_SOURCE_FACTS"


def claim_evidence_digest(
    kind: str, claim: Mapping[str, object], spans: tuple[EvidenceSpan, ...],
    assertion_context: AssertionContext | None = None,
) -> str:
    payload = {"kind": kind, "claim": dict(claim),
               "evidence": [span.model_dump(mode="json") for span in spans]}
    if assertion_context and any(assertion_context.model_dump().values()):
        payload["applicability_context"] = assertion_context.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def evidence_context_digest(document: ChunkedDocument, spans: tuple[EvidenceSpan, ...]) -> str:
    context = [
        {"chunk_index": span.chunk_index, "raw_text": document.chunks[span.chunk_index].text}
        for span in spans
    ]
    return hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_ENTITY_IDENTITY_KEYS = {
    NodeType.ORGANIZATION: ("name",),
    NodeType.APPLICATION: ("name",),
    NodeType.POLICY: ("title", "version"),
    NodeType.OBLIGATION: ("requirement_text",),
    NodeType.CONTROL: ("name",),
    NodeType.FINDING: ("description", "status"),
    NodeType.RISK: ("name",),
    NodeType.EVIDENCE: ("chunk_id",),
}


def stable_semantic_node_id(node_type: str | NodeType, properties: Mapping[str, object]) -> str:
    kind = node_type if isinstance(node_type, NodeType) else NodeType(node_type.upper())
    keys = _ENTITY_IDENTITY_KEYS.get(kind)
    if not keys:
        raise ValueError("Structural node identities are managed by structural ingestion")
    identity: dict[str, str] = {}
    for key in keys:
        value = properties.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Stable {kind.value} identity requires {key}")
        identity[key] = " ".join(value.casefold().split())
    digest = hashlib.sha256(
        json.dumps([kind.value, identity], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{kind.value.lower()}:{digest}"


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_type: NonEmptyStr
    properties: dict[str, object]
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    epistemic_mode: EpistemicMode = EpistemicMode.VERBATIM_EXTRACTION
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)
    support_attestation: EvidenceSupportAttestation | None = None
    assertion_context: AssertionContext = Field(default_factory=AssertionContext)

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
    support_attestation: EvidenceSupportAttestation | None = None
    assertion_context: AssertionContext = Field(default_factory=AssertionContext)

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
    extraction_provider: NonEmptyStr | None = None
    extraction_model: NonEmptyStr | None = None
    nodes: tuple[GraphNode, ...]
    relationships: tuple[GraphRelationship, ...]

    @model_validator(mode="after")
    def validate_before_write(self) -> Self:
        _validate_batch_fields(self)
        return self


def _validate_batch_fields(batch: GraphWriteBatch) -> None:
    if batch.screened.document != batch.document:
        raise ValueError("Screening results do not match the source document")
    if batch.nodes or batch.relationships:
        if not batch.extraction_provider or not batch.extraction_model:
            raise ValueError("Graph claims require extraction provider and model provenance")
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
        if node.confidence < 0.90:
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
        claim = {key: value for key, value in node.properties.items() if key != "id"}
        _validate_support_attestation(
            node.support_attestation,
            claim_evidence_digest(node_type, claim, node.evidence_spans, node.assertion_context),
            node.evidence_spans,
            evidence_context_digest(batch.document, node.evidence_spans),
        )
        if node.support_attestation is not None and node.support_attestation.model == batch.extraction_model:
            raise ValueError("Evidence verifier must be independent of the extraction model")
    for edge in batch.relationships:
        expected_relation_id = "relation:" + hashlib.sha256(
            json.dumps(
                [edge.source_id, edge.relation_type.upper(), edge.target_id],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if edge.id != expected_relation_id:
            raise ValueError("Relationship ID must match its deterministic endpoint identity")
        if edge.relation_type.upper() in {"HAS_CHUNK", "NEXT_CHUNK", "EXTRACTED_FROM"}:
            raise ValueError("Structural provenance relationships are managed by structural ingestion")
        if edge.epistemic_mode is not EpistemicMode.VERBATIM_EXTRACTION:
            raise ValueError("Inferred or rule-derived relationships are not admitted as graph facts")
        if edge.confidence < 0.90:
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
        _validate_support_attestation(
            edge.support_attestation,
            claim_evidence_digest(
                "RELATIONSHIP",
                {"source_id": edge.source_id, "relation_type": edge.relation_type.upper(),
                 "target_id": edge.target_id},
                edge.evidence_spans,
                edge.assertion_context,
            ),
            edge.evidence_spans,
            evidence_context_digest(batch.document, edge.evidence_spans),
        )
        if edge.support_attestation is not None and edge.support_attestation.model == batch.extraction_model:
            raise ValueError("Evidence verifier must be independent of the extraction model")


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
    if (VISUAL_FACT_EVIDENCE_POLICY == "DENY_IMAGE_VLM_SOURCE_FACTS"
            and any(source.extraction_method == "IMAGE_VLM" for source in chunk.sources)):
        raise ValueError("AI-generated visual descriptions cannot be admitted as original source facts")
    if any(
        source.extraction_method == "IMAGE_OCR" and source.source_image_sha256 is None
        for source in chunk.sources
    ):
        raise ValueError("OCR evidence is missing its source image SHA-256 provenance")


def _validate_support_attestation(
    attestation: EvidenceSupportAttestation | None,
    expected_claim_sha256: str,
    evidence_spans: tuple[EvidenceSpan, ...],
    expected_context_sha256: str,
) -> None:
    if attestation is None:
        raise ValueError("Every graph claim requires independent evidence-support verification")
    if attestation.claim_sha256 != expected_claim_sha256:
        raise ValueError("Evidence-support attestation does not match the proposed claim")
    if attestation.context_sha256 != expected_context_sha256:
        raise ValueError("Evidence-support attestation does not match the source chunk context")
    if not attestation.supported or not attestation.directly_stated:
        raise ValueError("Evidence verifier did not confirm direct source support")
    if attestation.confidence < 0.90:
        raise ValueError("Evidence verifier confidence is below the admission threshold")
    valid_quotes = {
        hashlib.sha256(span.quote.encode("utf-8")).hexdigest()
        for span in evidence_spans
    }
    if attestation.supporting_quote_sha256 not in valid_quotes:
        raise ValueError("Evidence verifier quote does not match a supplied source span")


def validate_graph_write_batch(batch: GraphWriteBatch) -> None:
    """Validate a batch before a caller executes any graph mutations."""
    if not isinstance(batch, GraphWriteBatch):
        raise TypeError("Expected a validated GraphWriteBatch")
    # Pydantic's frozen model is shallow; revalidate mutable property mappings.
    _validate_batch_fields(batch)
