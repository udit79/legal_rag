"""Transactional writer for GraphWriteGate-approved semantic graph batches."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.domain.ontology import NodeType, RelationType
from src.graph.graph_write_gate import GraphWriteBatch, validate_graph_write_batch

_TRACER = trace.get_tracer("graph_rag.graph.knowledge_writer")


class GraphIdentityConflict(RuntimeError):
    """A stable graph identity already exists with different immutable content."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _relationship_id(edge) -> str:
    if edge.id:
        return edge.id
    raw = _canonical([edge.source_id, edge.relation_type, edge.target_id])
    return "relation:" + hashlib.sha256(raw.encode()).hexdigest()


def _evidence_references(batch: GraphWriteBatch, spans) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for span in spans:
        chunk = batch.document.chunks[span.chunk_index]
        for source in chunk.sources:
            reference = {
                "source_document_id": batch.document.document_id,
                "ingestion_run_id": batch.ingestion_run_id,
                "chunk_index": span.chunk_index,
                "start_offset": span.start_offset,
                "end_offset": span.end_offset,
                "quote": span.quote,
                "page_number": source.page_number,
                "block_id": source.block_id,
            }
            references.append({key: value for key, value in reference.items() if value is not None})
    return references


def _node_query(node_type: str) -> str:
    label = NodeType(node_type.upper()).value
    return f"""
    UNWIND $rows AS row
    MERGE (n:{label} {{id: row.id}})
    ON CREATE SET n += row.properties
    RETURN row.id AS id, properties(n) AS properties
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
    RETURN row.id AS id,
           properties(r) AS properties,
           startNode(r).id AS source_id,
           endNode(r).id AS target_id
    """


async def _write_transaction(tx, batch: GraphWriteBatch) -> dict[str, int]:
    counts = {"nodes": 0, "relationships": 0}
    for node in batch.nodes:
        props = {
            **dict(node.properties),
            "source_document_id": batch.document.document_id,
            "ingestion_run_id": batch.ingestion_run_id,
            "evidence_references": _evidence_references(batch, node.evidence_spans),
        }
        expected_id = props["id"]
        rows = await tx.run(_node_query(node.node_type), rows=[{"id": expected_id, "properties": props}])
        records = [record async for record in rows]
        if len(records) != 1:
            raise GraphIdentityConflict("Graph node identity resolved to an unexpected record count")
        stored = dict(records[0]["properties"])
        if any(stored.get(k) != v for k, v in props.items()):
            raise GraphIdentityConflict(f"Existing {node.node_type} identity has conflicting properties")
        counts["nodes"] += 1
    for edge in batch.relationships:
        identity = _relationship_id(edge)
        node_types = {str(node.properties["id"]): node.node_type for node in batch.nodes}
        props = {
            "id": identity,
            "epistemic_mode": edge.epistemic_mode.value,
            "confidence": edge.confidence,
            "evidence_spans": [item.model_dump(mode="json") for item in edge.evidence_spans],
            "source_document_id": batch.document.document_id,
            "ingestion_run_id": batch.ingestion_run_id,
            "evidence_references": _evidence_references(batch, edge.evidence_spans),
        }
        rows = await tx.run(
            _relationship_query(edge.relation_type, node_types[edge.source_id], node_types[edge.target_id]),
            rows=[{"id": identity, "source_id": edge.source_id, "target_id": edge.target_id, "properties": props}],
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
    return counts


async def persist_approved_batch(db_connection, batch: GraphWriteBatch) -> dict[str, int]:
    """Only validated batches cross this boundary; writes are one Neo4j transaction."""
    with _TRACER.start_as_current_span("graph.knowledge.persist_approved_batch") as span:
        span.set_attribute("db.system", "neo4j")
        span.set_attribute("graph.write_gate.approved", True)
        try:
            validate_graph_write_batch(batch)
            if not batch.ingestion_run_id:
                raise ValueError("Approved knowledge graph writes require ingestion_run_id provenance")
            async with db_connection.get_session() as session:
                result = await session.execute_write(_write_transaction, batch)
            span.set_attribute("graph.nodes.processed", result["nodes"])
            span.set_attribute("graph.relationships.processed", result["relationships"])
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
