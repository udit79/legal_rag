"""Transactional writer for GraphWriteGate-approved semantic graph batches."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.domain.ontology import NodeType, RelationType
from src.graph.graph_write_gate import (
    EvidenceSupportAttestation,
    GraphWriteBatch,
    stable_semantic_node_id,
    validate_graph_write_batch,
)
from src.graph.assertion_contracts import (
    AssertionClassification,
    AssertionClassificationProposal,
    AssertionRecord,
    AssertionContext,
    applicability_overlaps,
    assertion_identity,
    classification_identity,
)
from src.graph.semantic_schema import semantic_constraints_ready

_TRACER = trace.get_tracer("graph_rag.graph.knowledge_writer")


class GraphIdentityConflict(RuntimeError):
    """A stable graph identity already exists with different immutable content."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _relationship_id(edge) -> str:
    raw = _canonical([edge.source_id, edge.relation_type, edge.target_id])
    return "relation:" + hashlib.sha256(raw.encode()).hexdigest()


_IDENTITY_KEYS = {
    "ORGANIZATION": ("name",),
    "APPLICATION": ("name",),
    "POLICY": ("title", "version"),
    "OBLIGATION": ("requirement_text",),
    "CONTROL": ("name",),
    "FINDING": ("description", "status"),
    "RISK": ("name",),
    "EVIDENCE": ("chunk_id",),
}


def _identity_properties(node_type: str, properties: dict[str, Any]) -> dict[str, Any]:
    keys = _IDENTITY_KEYS.get(node_type.upper())
    if keys is None:
        raise ValueError(f"Structural node type {node_type} cannot be written by semantic extraction")
    return {key: " ".join(str(properties[key]).casefold().split()) for key in keys}


def _attestation_payload(attestation: EvidenceSupportAttestation | None) -> dict[str, Any]:
    if attestation is None:
        raise GraphIdentityConflict("Graph write is missing independent evidence verification")
    return attestation.model_dump(mode="json")


def _evidence_references(batch: GraphWriteBatch, spans) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for span in spans:
        chunk = batch.document.chunks[span.chunk_index]
        for source in chunk.sources:
            source_provenance = source.model_dump(mode="json", exclude_defaults=True)
            source_provenance["extraction_method"] = source.extraction_method
            reference = {
                "source_document_id": batch.document.document_id,
                "ingestion_run_id": batch.ingestion_run_id,
                "chunk_index": span.chunk_index,
                "start_offset": span.start_offset,
                "end_offset": span.end_offset,
                "quote": span.quote,
                "page_number": source.page_number,
                "block_id": source.block_id,
                "source_sha256": batch.document.document_id,
                "source_provenance": source_provenance,
            }
            references.append({key: value for key, value in reference.items() if value is not None})
    return references


def _node_query(node_type: str) -> str:
    label = NodeType(node_type.upper()).value
    return f"""
    UNWIND $rows AS row
    MERGE (n:SEMANTIC_ENTITY {{id: row.id}})
    ON CREATE SET n += row.properties
    SET n:{label}, n.confidence = CASE
            WHEN n.confidence IS NULL OR row.confidence > n.confidence
                THEN row.confidence ELSE n.confidence END,
        n.epistemic_mode = row.epistemic_mode,
        n.source_document_ids = reduce(acc = coalesce(n.source_document_ids, []), item IN row.source_document_ids |
            CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
        n.source_sha256s = reduce(acc = coalesce(n.source_sha256s, []), item IN row.source_sha256s |
            CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
        n.ingestion_run_ids = reduce(acc = coalesce(n.ingestion_run_ids, []), item IN row.ingestion_run_ids |
            CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
        n.evidence_references = reduce(acc = coalesce(n.evidence_references, []), item IN row.evidence_references |
            CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
        n.extraction_assertions = reduce(acc = coalesce(n.extraction_assertions, []), item IN row.assertions |
            CASE WHEN item IN acc THEN acc ELSE acc + [item] END)
    RETURN row.id AS id, properties(n) AS properties, labels(n) AS labels
    """


def _relationship_query(relation_type: str, source_type: str, target_type: str) -> str:
    rel = RelationType(relation_type.upper()).value
    source_label = NodeType(source_type.upper()).value
    target_label = NodeType(target_type.upper()).value
    return f"""
    UNWIND $rows AS row
    MATCH (s:{source_label} {{id: row.source_id}}), (t:{target_label} {{id: row.target_id}})
    MERGE (s)-[r:{rel} {{id: row.id}}]->(t)
    ON CREATE SET r += row.properties
    SET r.confidence = CASE
            WHEN r.confidence IS NULL OR row.confidence > r.confidence
                THEN row.confidence ELSE r.confidence END,
        r.epistemic_mode = row.epistemic_mode,
        r.source_document_ids = reduce(acc = coalesce(r.source_document_ids, []), item IN row.source_document_ids |
            CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
        r.source_sha256s = reduce(acc = coalesce(r.source_sha256s, []), item IN row.source_sha256s |
            CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
        r.ingestion_run_ids = reduce(acc = coalesce(r.ingestion_run_ids, []), item IN row.ingestion_run_ids |
            CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
        r.evidence_references = reduce(acc = coalesce(r.evidence_references, []), item IN row.evidence_references |
            CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
        r.extraction_assertions = reduce(acc = coalesce(r.extraction_assertions, []), item IN row.assertions |
            CASE WHEN item IN acc THEN acc ELSE acc + [item] END)
    RETURN row.id AS id,
           properties(r) AS properties,
           startNode(r).id AS source_id,
           endNode(r).id AS target_id
    """


def _relationship_identity_query(relation_type: str) -> str:
    rel = RelationType(relation_type.upper()).value
    return f"""
    UNWIND $rows AS row
    OPTIONAL MATCH ()-[r:{rel} {{id: row.id}}]->()
    RETURN row.id AS id,
           collect(CASE WHEN r IS NULL THEN null ELSE {{
               source_id: startNode(r).id,
               target_id: endNode(r).id,
               properties: properties(r)
           }} END) AS matches
    """


_ASSERTION_CANDIDATES_QUERY = """
MATCH (a:KNOWLEDGE_ASSERTION {subject_id: $subject_id, predicate: $predicate})
WHERE a.id <> $id AND a.object_value <> $object_value
RETURN properties(a) AS properties
"""
_ASSERTION_WRITE_QUERY = """
MERGE (a:KNOWLEDGE_ASSERTION {id: $id})
ON CREATE SET a += $properties
SET a.source_sha256s = reduce(acc = coalesce(a.source_sha256s, []), item IN $source_sha256s |
        CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
    a.ingestion_run_ids = reduce(acc = coalesce(a.ingestion_run_ids, []), item IN $ingestion_run_ids |
        CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
    a.evidence_references = reduce(acc = coalesce(a.evidence_references, []), item IN $evidence_references |
        CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
    a.support_attestations = reduce(acc = coalesce(a.support_attestations, []), item IN $support_attestations |
        CASE WHEN item IN acc THEN acc ELSE acc + [item] END)
RETURN properties(a) AS properties
"""
_ASSERTION_CLASSIFICATION_IDENTITY_QUERY = """
OPTIONAL MATCH (:KNOWLEDGE_ASSERTION)-[r:ASSERTION_CLASSIFICATION {id: $id}]->(:KNOWLEDGE_ASSERTION)
RETURN collect(CASE WHEN r IS NULL THEN null ELSE {
    source_id: startNode(r).id, target_id: endNode(r).id, properties: properties(r)
} END) AS matches
"""
_ASSERTION_PAIR_PROVENANCE_QUERY = """
MATCH (a:KNOWLEDGE_ASSERTION)
WHERE a.id IN $ids
RETURN a.id AS id, a.source_sha256s AS source_sha256s,
       a.ingestion_run_ids AS ingestion_run_ids,
       a.evidence_references AS evidence_references
"""
_ASSERTION_CLASSIFICATION_WRITE_QUERY = """
MATCH (source:KNOWLEDGE_ASSERTION {id: $source_id}),
      (target:KNOWLEDGE_ASSERTION {id: $target_id})
MERGE (source)-[r:ASSERTION_CLASSIFICATION {id: $id}]->(target)
ON CREATE SET r += $properties
SET r.source_sha256s = reduce(acc = coalesce(r.source_sha256s, []), item IN $source_sha256s |
        CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
    r.ingestion_run_ids = reduce(acc = coalesce(r.ingestion_run_ids, []), item IN $ingestion_run_ids |
        CASE WHEN item IN acc THEN acc ELSE acc + [item] END),
    r.evidence_references = reduce(acc = coalesce(r.evidence_references, []), item IN $evidence_references |
        CASE WHEN item IN acc THEN acc ELSE acc + [item] END)
RETURN properties(r) AS properties, source.id AS source_id, target.id AS target_id
"""


def _assertion_record(*, subject_id: str, predicate: str, object_value: str,
                      context: AssertionContext) -> AssertionRecord:
    return AssertionRecord(
        id=assertion_identity(
            subject_id=subject_id, predicate=predicate, object_value=object_value, context=context
        ),
        subject_id=subject_id, predicate=predicate,
        object_value=object_value, context=context,
    )


def _batch_assertions(batch: GraphWriteBatch):
    for node in batch.nodes:
        properties = {key: value for key, value in node.properties.items() if key != "id"}
        yield (
            _assertion_record(
                subject_id=str(node.properties["id"]), predicate="ENTITY_PROPERTIES",
                object_value=_canonical(properties), context=node.assertion_context,
            ),
            _evidence_references(batch, node.evidence_spans),
            _attestation_payload(node.support_attestation),
        )
    for edge in batch.relationships:
        yield (
            _assertion_record(
                subject_id=edge.source_id, predicate=edge.relation_type.upper(),
                object_value=edge.target_id, context=edge.assertion_context,
            ),
            _evidence_references(batch, edge.evidence_spans),
            _attestation_payload(edge.support_attestation),
        )


def _claims_conflict(existing: dict, assertion: AssertionRecord) -> bool:
    if existing.get("predicate") != assertion.predicate:
        return False
    if assertion.predicate == "ENTITY_PROPERTIES":
        try:
            old = json.loads(existing["object_value"])
            new = json.loads(assertion.object_value)
        except (TypeError, ValueError):
            return True
        return any(old[key] != new[key] for key in old.keys() & new.keys())
    return existing.get("object_value") != assertion.object_value


async def _persist_classification_tx(tx, proposal: AssertionClassificationProposal) -> None:
    provenance_result = await tx.run(
        _ASSERTION_PAIR_PROVENANCE_QUERY,
        ids=[proposal.source_assertion_id, proposal.target_assertion_id],
    )
    provenance_records = [dict(record) async for record in provenance_result]
    if len(provenance_records) != 2:
        raise GraphIdentityConflict("Classification endpoints must be persisted assertions")
    source_hashes = {
        item for record in provenance_records for item in record.get("source_sha256s", [])
    }
    ingestion_runs = {
        item for record in provenance_records for item in record.get("ingestion_run_ids", [])
    }
    evidence = {
        item for record in provenance_records for item in record.get("evidence_references", [])
    }
    if (not set(proposal.source_sha256s).issubset(source_hashes)
            or not set(proposal.ingestion_run_ids).issubset(ingestion_runs)
            or not set(proposal.evidence_references).issubset(evidence)):
        raise GraphIdentityConflict("Classification provenance must reference evidence on its assertions")
    identity = classification_identity(proposal)
    identity_result = await tx.run(_ASSERTION_CLASSIFICATION_IDENTITY_QUERY, id=identity)
    identity_records = [record async for record in identity_result]
    if len(identity_records) != 1:
        raise GraphIdentityConflict("Assertion classification identity lookup was ambiguous")
    existing = identity_records[0]["matches"]
    if len(existing) > 1:
        raise GraphIdentityConflict("Duplicate assertion classification identity already exists")
    properties = {
        "id": identity,
        "classification": proposal.classification.value,
        "classifier": proposal.classifier,
        "rationale": proposal.rationale,
        "adjudication_status": proposal.adjudication_status,
    }
    if existing:
        current = existing[0]
        if (current["source_id"] != proposal.source_assertion_id
                or current["target_id"] != proposal.target_assertion_id
                or any(current["properties"].get(key) != value for key, value in properties.items())):
            raise GraphIdentityConflict("Assertion classification identity has conflicting content")
    result = await tx.run(
        _ASSERTION_CLASSIFICATION_WRITE_QUERY,
        id=identity, source_id=proposal.source_assertion_id,
        target_id=proposal.target_assertion_id, properties=properties,
        source_sha256s=list(proposal.source_sha256s),
        ingestion_run_ids=list(proposal.ingestion_run_ids),
        evidence_references=list(proposal.evidence_references),
    )
    records = [record async for record in result]
    if len(records) != 1:
        raise GraphIdentityConflict("Assertion classification endpoints are missing or ambiguous")
    stored = dict(records[0]["properties"])
    if (records[0]["source_id"] != proposal.source_assertion_id
            or records[0]["target_id"] != proposal.target_assertion_id
            or any(stored.get(key) != value for key, value in properties.items())):
        raise GraphIdentityConflict("Assertion classification has conflicting endpoints or content")


async def _persist_assertion_tx(tx, batch: GraphWriteBatch, assertion: AssertionRecord,
                                evidence_references: list[dict[str, Any]], attestation: dict[str, Any]) -> None:
    context = assertion.context.model_dump(mode="json")
    rows = await tx.run(
        _ASSERTION_CANDIDATES_QUERY,
        id=assertion.id,
        subject_id=assertion.subject_id,
        predicate=assertion.predicate,
        object_value=assertion.object_value,
    )
    candidates = [dict(record["properties"]) async for record in rows]

    properties = {
        "id": assertion.id,
        "subject_id": assertion.subject_id,
        "predicate": assertion.predicate,
        "object_value": assertion.object_value,
        **context,
    }
    result = await tx.run(
        _ASSERTION_WRITE_QUERY,
        id=assertion.id,
        properties=properties,
        source_sha256s=[batch.document.document_id],
        ingestion_run_ids=[batch.ingestion_run_id],
        evidence_references=[_canonical(reference) for reference in evidence_references],
        support_attestations=[_canonical(attestation)],
    )
    records = [record async for record in result]
    if len(records) != 1:
        raise GraphIdentityConflict("Knowledge assertion identity resolved to an unexpected record count")
    stored = dict(records[0]["properties"])
    if any(stored.get(key) != value for key, value in properties.items()):
        raise GraphIdentityConflict("Existing knowledge assertion has conflicting immutable content")

    for existing in candidates:
        if (not applicability_overlaps(existing, assertion.context)
                or not _claims_conflict(existing, assertion)):
            continue
        # UNRESOLVED is a conservative conflict candidate, never a legal verdict.
        source_id, target_id = sorted((assertion.id, str(existing["id"])))
        old_evidence = list(existing.get("evidence_references") or [])
        classification = AssertionClassificationProposal(
            source_assertion_id=source_id,
            target_assertion_id=target_id,
            classification=AssertionClassification.UNRESOLVED,
            classifier="deterministic_overlap_scan",
            rationale=("Different asserted values share subject, predicate, jurisdiction, scope, "
                       "and potentially overlapping validity; no truth adjudication was performed."),
            source_sha256s=tuple(sorted({
                *stored.get("source_sha256s", []), *existing.get("source_sha256s", [])
            })),
            ingestion_run_ids=tuple(sorted({
                *stored.get("ingestion_run_ids", []), *existing.get("ingestion_run_ids", [])
            })),
            evidence_references=tuple(dict.fromkeys([
                *old_evidence, *stored.get("evidence_references", [])
            ])),
        )
        await _persist_classification_tx(tx, classification)


async def _write_transaction(tx, batch: GraphWriteBatch) -> dict[str, int]:
    if not await semantic_constraints_ready(tx):
        raise RuntimeError(
            "Semantic uniqueness constraints are missing; run the explicit semantic schema "
            "preflight/bootstrap before graph writes"
        )
    counts = {"nodes": 0, "relationships": 0}
    for node in batch.nodes:
        props = {
            **dict(node.properties),
        }
        expected_id = props["id"]
        if expected_id != stable_semantic_node_id(node.node_type, props):
            raise GraphIdentityConflict("Semantic node ID does not match its cross-document identity")
        identity_properties = _identity_properties(node.node_type, props)
        references = _evidence_references(batch, node.evidence_spans)
        assertion = {
            "properties": {key: value for key, value in props.items() if key != "id"},
            "confidence": node.confidence,
            "epistemic_mode": node.epistemic_mode.value,
            "support_attestation": _attestation_payload(node.support_attestation),
            "extraction_provider": batch.extraction_provider,
            "extraction_model": batch.extraction_model,
            "evidence_references": references,
        }
        reference_values = [_canonical(reference) for reference in references]
        assertion_values = [_canonical(assertion)]
        rows = await tx.run(_node_query(node.node_type), rows=[{
            "id": expected_id, "properties": props,
            "confidence": node.confidence, "epistemic_mode": node.epistemic_mode.value,
            "source_document_ids": [batch.document.document_id],
            "source_sha256s": [batch.document.document_id],
            "ingestion_run_ids": [batch.ingestion_run_id],
            "evidence_references": reference_values, "assertions": assertion_values,
        }])
        records = [record async for record in rows]
        if len(records) != 1:
            raise GraphIdentityConflict("Graph node identity resolved to an unexpected record count")
        stored = dict(records[0]["properties"])
        labels = set(records[0].get("labels", ()))
        try:
            stored_identity = _identity_properties(node.node_type, stored)
        except (KeyError, ValueError):
            stored_identity = {}
        if stored_identity != identity_properties or node.node_type.upper() not in labels:
            raise GraphIdentityConflict(f"Existing {node.node_type} identity has conflicting identity properties")
        counts["nodes"] += 1
    for edge in batch.relationships:
        identity = _relationship_id(edge)
        identity_result = await tx.run(
            _relationship_identity_query(edge.relation_type), rows=[{"id": identity}]
        )
        identity_rows = [record async for record in identity_result]
        if len(identity_rows) != 1:
            raise GraphIdentityConflict("Relationship identity lookup was ambiguous")
        existing_matches = identity_rows[0]["matches"]
        if len(existing_matches) > 1:
            raise GraphIdentityConflict("Duplicate relationship identity already exists")
        if existing_matches:
            existing = existing_matches[0]
            if existing["source_id"] != edge.source_id or existing["target_id"] != edge.target_id:
                raise GraphIdentityConflict("Existing relationship identity has incorrect endpoints")
        node_types = {str(node.properties["id"]): node.node_type for node in batch.nodes}
        references = _evidence_references(batch, edge.evidence_spans)
        props = {"id": identity}
        assertion = {
            "confidence": edge.confidence,
            "epistemic_mode": edge.epistemic_mode.value,
            "support_attestation": _attestation_payload(edge.support_attestation),
            "extraction_provider": batch.extraction_provider,
            "extraction_model": batch.extraction_model,
            "evidence_references": references,
        }
        reference_values = [_canonical(reference) for reference in references]
        assertion_values = [_canonical(assertion)]
        rows = await tx.run(
            _relationship_query(edge.relation_type, node_types[edge.source_id], node_types[edge.target_id]),
            rows=[{
                "id": identity, "source_id": edge.source_id, "target_id": edge.target_id,
                "properties": props, "confidence": edge.confidence,
                "epistemic_mode": edge.epistemic_mode.value,
                "source_document_ids": [batch.document.document_id],
                "source_sha256s": [batch.document.document_id],
                "ingestion_run_ids": [batch.ingestion_run_id],
                "evidence_references": reference_values, "assertions": assertion_values,
            }],
        )
        records = [record async for record in rows]
        if len(records) != 1:
            raise GraphIdentityConflict("Relationship endpoints are missing or ambiguous")
        record = records[0]
        stored = dict(record["properties"])
        if (record["source_id"] != edge.source_id or record["target_id"] != edge.target_id
                or any(stored.get(k) != v for k, v in props.items())):
            raise GraphIdentityConflict("Existing relationship identity has conflicting endpoints or properties")
        counts["relationships"] += 1
    for assertion, references, attestation in _batch_assertions(batch):
        await _persist_assertion_tx(tx, batch, assertion, references, attestation)
    return counts


async def _persist_assertion_classification_transaction(tx, proposal: AssertionClassificationProposal) -> None:
    if not await semantic_constraints_ready(tx):
        raise RuntimeError("Semantic uniqueness constraints must be initialized before classification writes")
    await _persist_classification_tx(tx, proposal)


async def persist_assertion_classification(db_connection, proposal: AssertionClassificationProposal) -> None:
    """Persist a sourced, explicitly proposed classification without adjudicating it."""
    async with db_connection.get_session() as session:
        await session.execute_write(_persist_assertion_classification_transaction, proposal)


async def persist_approved_batch(db_connection, batch: GraphWriteBatch) -> dict[str, int]:
    """Only validated batches cross this boundary; writes are one Neo4j transaction."""
    with _TRACER.start_as_current_span("graph.knowledge.persist_approved_batch") as span:
        span.set_attribute("db.system", "neo4j")
        span.set_attribute("graph.write_gate.approved", True)
        try:
            validate_graph_write_batch(batch)
            if not batch.ingestion_run_id:
                raise ValueError("Approved knowledge graph writes require ingestion_run_id provenance")
            if not batch.extraction_provider or not batch.extraction_model:
                raise ValueError("Approved knowledge graph writes require extraction model provenance")
            if any(
                node.support_attestation is None or node.support_attestation.model == batch.extraction_model
                for node in batch.nodes
            ) or any(
                edge.support_attestation is None or edge.support_attestation.model == batch.extraction_model
                for edge in batch.relationships
            ):
                raise ValueError("Every claim requires an independent evidence verifier")
            async with db_connection.get_session() as session:
                result = await session.execute_write(_write_transaction, batch)
            span.set_attribute("graph.nodes.processed", result["nodes"])
            span.set_attribute("graph.relationships.processed", result["relationships"])
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
