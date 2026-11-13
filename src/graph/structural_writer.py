"""Deterministic, fail-closed persistence for the admitted structural corpus."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.graph.corpus_reader import CorpusDocument

_TRACER = trace.get_tracer("graph_rag.graph.structural_writer")

_DOCUMENT_QUERY = """
UNWIND $rows AS row
OPTIONAL MATCH (d:DOCUMENT {id: row.id})
RETURN row.id AS id, properties(d) AS existing
"""
_CHUNK_QUERY = """
UNWIND $rows AS row
OPTIONAL MATCH (c:CHUNK {id: row.id})
RETURN row.id AS id, properties(c) AS existing
"""
_HAS_CHUNK_CONFLICT_QUERY = """
UNWIND $rows AS row
OPTIONAL MATCH ()-[r:HAS_CHUNK {id: row.id}]->()
WITH row, collect(r) AS matches
RETURN row.id AS id,
       size(matches) AS match_count,
       [rel IN matches | {
           properties: properties(rel),
           source_id: startNode(rel).id,
           target_id: endNode(rel).id
       }] AS matches
"""
_NEXT_CHUNK_CONFLICT_QUERY = """
UNWIND $rows AS row
OPTIONAL MATCH ()-[r:NEXT_CHUNK {id: row.id}]->()
WITH row, collect(r) AS matches
RETURN row.id AS id,
       size(matches) AS match_count,
       [rel IN matches | {
           properties: properties(rel),
           source_id: startNode(rel).id,
           target_id: endNode(rel).id
       }] AS matches
"""
_LOCK_DOCUMENT_QUERY = """
UNWIND $rows AS row
MERGE (d:DOCUMENT {id: row.id})
ON CREATE SET d += row
ON MATCH SET d.graph_write_lock = coalesce(d.graph_write_lock, 0) + 1
RETURN d.id AS id,
       d.current_ingestion_run_id AS current_run_id,
       d.current_ingestion_created_at AS current_created_at,
       d.current_chunking_policy_version AS current_policy_version
"""
_UPDATE_DOCUMENT_CURRENT_QUERY = """
UNWIND $rows AS row
MATCH (d:DOCUMENT {id: row.id})
SET d.current_chunking_policy_version = row.current_chunking_policy_version,
    d.current_ingestion_run_id = row.current_ingestion_run_id,
    d.current_ingestion_created_at = row.current_ingestion_created_at
"""
_WRITE_CHUNK_QUERY = """
UNWIND $rows AS row
MERGE (c:CHUNK {id: row.id})
ON CREATE SET c += row
"""
_HAS_CHUNK_QUERY = """
UNWIND $rows AS row
MATCH (d:DOCUMENT {id: row.source_sha256})
MATCH (c:CHUNK {id: row.chunk_id})
MERGE (d)-[r:HAS_CHUNK {id: row.id}]->(c)
ON CREATE SET r.source_sha256 = row.source_sha256,
              r.chunk_id = row.chunk_id,
              r.ingestion_run_history = [row.run_history_entry]
ON MATCH SET r.ingestion_run_history =
    CASE
        WHEN row.run_history_entry IN coalesce(r.ingestion_run_history, [])
            THEN coalesce(r.ingestion_run_history, [])
        ELSE coalesce(r.ingestion_run_history, []) + [row.run_history_entry]
    END
"""
_NEXT_CHUNK_QUERY = """
UNWIND $rows AS row
MATCH (a:CHUNK {id: row.from_id})
MATCH (b:CHUNK {id: row.to_id})
MERGE (a)-[r:NEXT_CHUNK {id: row.id}]->(b)
ON CREATE SET r.source_sha256 = row.source_sha256,
              r.from_chunk_id = row.from_id,
              r.to_chunk_id = row.to_id,
              r.ingestion_run_id = row.ingestion_run_id,
              r.chunking_policy_version = row.chunking_policy_version
"""
_MARK_HISTORICAL_CHUNKS_QUERY = """
UNWIND $rows AS row
MATCH (d:DOCUMENT {id: row.id})-[:HAS_CHUNK]->(c:CHUNK)
SET c.is_current = false
"""
_MARK_CURRENT_CHUNKS_QUERY = """
UNWIND $rows AS row
MATCH (d:DOCUMENT {id: row.source_sha256})-[:HAS_CHUNK]->(c:CHUNK {id: row.chunk_id})
SET c.is_current = true
"""

# Retrieval must filter both current markers. Old versions and chunks omitted
# from the latest run remain available for audit, but cannot enter retrieval.
CURRENT_CHUNK_RETRIEVAL_QUERY = """
CYPHER 25
MATCH (c:CHUNK)
SEARCH c IN (
  FULLTEXT INDEX chunk_retrieval_text_fulltext
  FOR $query
  LIMIT $candidate_count
) SCORE AS score
MATCH (d:DOCUMENT)-[:HAS_CHUNK]->(c)
WHERE c.is_current = true
  AND c.source_sha256 = d.source_sha256
  AND c.chunking_policy_version = d.current_chunking_policy_version
  AND d.current_ingestion_run_id IS NOT NULL
RETURN d, c, score
ORDER BY score DESC, c.id ASC
LIMIT $limit
"""
CURRENT_NEXT_CHUNK_QUERY = """
MATCH (d:DOCUMENT)-[:HAS_CHUNK]->(a:CHUNK)
MATCH (d)-[:HAS_CHUNK]->(b:CHUNK)
MATCH (a)-[r:NEXT_CHUNK]->(b)
WHERE a.is_current = true
  AND b.is_current = true
  AND a.chunking_policy_version = d.current_chunking_policy_version
  AND b.chunking_policy_version = d.current_chunking_policy_version
  AND r.source_sha256 = d.source_sha256
  AND r.ingestion_run_id = d.current_ingestion_run_id
  AND r.chunking_policy_version = d.current_chunking_policy_version
RETURN d, a, r, b
"""


class GraphIdentityConflict(ValueError):
    pass


def validate_structural_document(document: CorpusDocument) -> None:
    if len(document.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in document.source_sha256):
        raise ValueError("Invalid document source SHA-256")
    if not document.ingestion_run_id.strip() or not document.chunking_policy_version.strip():
        raise ValueError("Run ID and chunking policy are required")
    if (not isinstance(document.ingestion_created_at, datetime)
            or document.ingestion_created_at.tzinfo is None
            or document.ingestion_created_at.utcoffset() is None):
        raise ValueError("Ingestion created_at must be a timezone-aware datetime")
    if (len(document.canonical_document_digest) != 64
            or any(c not in "0123456789abcdef" for c in document.canonical_document_digest)
            or type(document.source_page_count) is not int
            or document.source_page_count <= 0):
        raise ValueError("Canonical document provenance is invalid")
    if not document.screening_policy_version.strip() or len(document.screening_policy_digest) != 64 or any(
        character not in "0123456789abcdef" for character in document.screening_policy_digest
    ):
        raise ValueError("Screening policy provenance is invalid")
    if not document.chunks:
        raise ValueError("CPCR structural output must contain at least one admitted chunk")
    final_indices = [chunk.chunk_index for chunk in document.chunks]
    if final_indices != sorted(set(final_indices)):
        raise ValueError("Final chunk indices must be ordered and unique")
    chunk_ids = [chunk.id for chunk in document.chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise GraphIdentityConflict(
            "Multiple final chunks share the same deterministic source identity"
        )
    for chunk in document.chunks:
        if not chunk.original_chunk_indices or tuple(sorted(set(chunk.original_chunk_indices))) != chunk.original_chunk_indices:
            raise ValueError("Original source indexes must be ordered and unique")
        if any(decision not in {"ADMITTED", "ADMITTED_FLAGGED"} for decision in chunk.admission_decisions):
            raise ValueError("Quarantined or rejected chunks cannot enter the graph")
        if not chunk.raw_text or not chunk.retrieval_text or not chunk.sources:
            raise ValueError("Graph chunks require text and source provenance")


def _doc_properties(document: CorpusDocument) -> dict[str, Any]:
    return {
        "id": document.source_sha256,
        "source_sha256": document.source_sha256,
        "canonical_document_digest": document.canonical_document_digest,
        "source_page_count": document.source_page_count,
        "current_chunking_policy_version": document.chunking_policy_version,
        "current_ingestion_run_id": document.ingestion_run_id,
        "current_ingestion_created_at": document.ingestion_created_at.astimezone(timezone.utc).isoformat(),
    }


def _doc_identity_properties(document: CorpusDocument) -> dict[str, Any]:
    return {
        "id": document.source_sha256,
        "source_sha256": document.source_sha256,
        "canonical_document_digest": document.canonical_document_digest,
        "source_page_count": document.source_page_count,
    }


def _chunk_properties(document: CorpusDocument, chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "source_sha256": document.source_sha256,
        "chunking_policy_version": document.chunking_policy_version,
        "original_chunk_indices": list(chunk.original_chunk_indices),
        "original_chunk_indices_key": ",".join(map(str, chunk.original_chunk_indices)),
        "raw_text": chunk.raw_text,
        "retrieval_text": chunk.retrieval_text,
        "chunk_sha256": chunk.chunk_sha256,
        "chunk_index": chunk.chunk_index,
        "sources_json": json.dumps(chunk.sources, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")),
        "retrieval_text_edits_json": json.dumps(
            chunk.retrieval_text_edits, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")),
    }


def _record_value(record: Any, key: str):
    try:
        return record[key]
    except (KeyError, TypeError):
        return getattr(record, key)


async def _assert_no_conflicts(tx, query: str, rows: list[dict[str, Any]]) -> None:
    result = await tx.run(query, rows=rows)
    records = await result.data()
    for record in records:
        existing = _record_value(record, "existing")
        if existing is None:
            continue
        expected = next(row for row in rows if row["id"] == _record_value(record, "id"))
        current = dict(existing)
        if any(current.get(key) != value for key, value in expected.items()):
            raise GraphIdentityConflict(
                f"Existing graph identity {_record_value(record, 'id')} has conflicting content"
            )


async def _assert_has_chunk_identities(tx, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    result = await tx.run(_HAS_CHUNK_CONFLICT_QUERY, rows=rows)
    for record in await result.data():
        identity = _record_value(record, "id")
        expected = next(row for row in rows if row["id"] == identity)
        match_count = _record_value(record, "match_count")
        if match_count > 1:
            raise GraphIdentityConflict(
                f"HAS_CHUNK identity {identity} exists more than once"
            )
        if match_count == 1:
            matches = _record_value(record, "matches")
            match = matches[0]
            properties = match["properties"]
            immutable_properties = {
                key: expected[key] for key in ("id", "source_sha256", "chunk_id")
            }
            if (match["source_id"] != expected["source_sha256"]
                    or match["target_id"] != expected["chunk_id"]
                    or any(properties.get(key) != value
                           for key, value in immutable_properties.items())):
                raise GraphIdentityConflict(
                    f"HAS_CHUNK identity {identity} has conflicting properties or endpoints"
                )


async def _assert_next_chunk_identities(tx, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    result = await tx.run(_NEXT_CHUNK_CONFLICT_QUERY, rows=rows)
    for record in await result.data():
        identity = _record_value(record, "id")
        expected = next(row for row in rows if row["id"] == identity)
        match_count = _record_value(record, "match_count")
        if match_count > 1:
            raise GraphIdentityConflict(
                f"NEXT_CHUNK identity {identity} exists more than once"
            )
        if match_count == 1:
            match = _record_value(record, "matches")[0]
            properties = match["properties"]
            immutable_properties = {
                key: expected[key]
                for key in (
                    "id",
                    "source_sha256",
                    "from_chunk_id",
                    "to_chunk_id",
                    "ingestion_run_id",
                    "chunking_policy_version",
                )
            }
            if (
                match["source_id"] != expected["from_id"]
                or match["target_id"] != expected["to_id"]
                or any(
                    properties.get(key) != value
                    for key, value in immutable_properties.items()
                )
            ):
                raise GraphIdentityConflict(
                    f"NEXT_CHUNK identity {identity} has conflicting properties or endpoints"
                )


def _reject_stale_run(
    document: CorpusDocument,
    current_run_id: str | None,
    current_created_at: str | datetime | None,
    current_policy_version: str | None = None,
) -> None:
    """Reject stale or internally inconsistent ingestion snapshots.

    The same run is idempotent only when its authoritative timestamp also
    matches. Different runs must have strictly increasing timestamps; equal
    timestamps are ambiguous and are rejected rather than tie-broken by ID.
    """
    incoming_time = document.ingestion_created_at
    if incoming_time.tzinfo is None or incoming_time.utcoffset() is None:
        raise GraphIdentityConflict("Incoming ingestion timestamp has no timezone")
    incoming_time_utc = incoming_time.astimezone(timezone.utc)

    if current_run_id is None:
        if current_created_at is not None or current_policy_version is not None:
            raise GraphIdentityConflict(
                "Existing document has current-ingestion metadata but no current run ID"
            )
        return

    if not isinstance(current_run_id, str) or not current_run_id.strip():
        raise GraphIdentityConflict("Existing current ingestion run ID is invalid")
    if current_created_at is None:
        raise GraphIdentityConflict(
            "Existing current ingestion has no timestamp; refusing an unordered replacement"
        )

    try:
        if isinstance(current_created_at, datetime):
            existing_time = current_created_at
        elif isinstance(current_created_at, str):
            existing_time = datetime.fromisoformat(
                current_created_at.replace("Z", "+00:00")
            )
        else:
            raise TypeError("Unsupported timestamp value")
    except (TypeError, ValueError) as exc:
        raise GraphIdentityConflict(
            "Existing current ingestion timestamp is invalid"
        ) from exc

    if existing_time.tzinfo is None or existing_time.utcoffset() is None:
        raise GraphIdentityConflict(
            "Existing current ingestion timestamp has no timezone"
        )
    existing_time_utc = existing_time.astimezone(timezone.utc)

    if document.ingestion_run_id == current_run_id:
        if incoming_time_utc != existing_time_utc:
            raise GraphIdentityConflict(
                "Same ingestion run ID has inconsistent timestamps"
            )
        if current_policy_version != document.chunking_policy_version:
            raise GraphIdentityConflict(
                "Same ingestion run ID has an inconsistent chunking policy version"
            )
        return

    if incoming_time_utc < existing_time_utc:
        raise GraphIdentityConflict(
            f"Stale ingestion run {document.ingestion_run_id} "
            f"cannot replace current run {current_run_id}"
        )
    if incoming_time_utc == existing_time_utc:
        raise GraphIdentityConflict(
            "Ambiguous ingestion ordering: different run IDs have equal timestamps"
        )


async def _run(tx, query: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    result = await tx.run(query, rows=rows)
    summary = await result.consume()
    return summary.counters.nodes_created + summary.counters.relationships_created


class StructuralGraphWriter:
    async def write(self, db_connection, documents: tuple[CorpusDocument, ...]) -> dict[str, int]:
        with _TRACER.start_as_current_span("graph.neo4j.write_structural_corpus") as span:
            span.set_attribute("db.system", "neo4j")
            database = getattr(db_connection, "database", None)
            if isinstance(database, str) and database:
                span.set_attribute("db.namespace", database)
            span.set_attribute("graph.document.count", len(documents))
            span.set_attribute("graph.ingestion.run_ids", [doc.ingestion_run_id for doc in documents])
            span.set_attribute("graph.document.source_sha256s", [doc.source_sha256 for doc in documents])
            span.set_attribute("graph.chunk.count", sum(len(doc.chunks) for doc in documents))
            try:
                counts = await self._write(db_connection, documents)
                for key, value in counts.items():
                    span.set_attribute(f"graph.write.{key}", value)
                return counts
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise

    async def _write(self, db_connection, documents: tuple[CorpusDocument, ...]) -> dict[str, int]:
        if not documents:
            raise ValueError("No corpus documents to write")
        for document in documents:
            validate_structural_document(document)
        if len({document.source_sha256 for document in documents}) != len(documents):
            raise ValueError("Corpus contains duplicate source documents")

        doc_rows = [_doc_properties(document) for document in documents]
        doc_identity_rows = [_doc_identity_properties(document) for document in documents]
        chunk_rows = [
            _chunk_properties(document, chunk)
            for document in documents for chunk in document.chunks
        ]
        has_chunk_rows = [
            {
                "id": hashlib.sha256(
                    f"{document.source_sha256}\0{chunk.id}".encode("utf-8")
                ).hexdigest(),
                "source_sha256": document.source_sha256,
                "chunk_id": chunk.id,
                "run_history_entry": json.dumps(
                    {
                        "ingestion_run_id": document.ingestion_run_id,
                        "chunking_policy_version": document.chunking_policy_version,
                        "screening_policy_version": document.screening_policy_version,
                        "screening_policy_digest": document.screening_policy_digest,
                        "admission_decisions": chunk.admission_decisions,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            for document in documents for chunk in document.chunks
        ]
        next_rows = []
        for document in documents:
            for left, right in zip(document.chunks, document.chunks[1:]):
                next_original_index = left.original_chunk_indices[-1] + 1
                if next_original_index in right.original_chunk_indices:
                    next_rows.append({
                        "id": hashlib.sha256(
                            f"{document.source_sha256}\0{left.id}\0{right.id}\0{document.ingestion_run_id}".encode("utf-8")
                        ).hexdigest(),
                        "source_sha256": document.source_sha256,
                        "from_id": left.id,
                        "to_id": right.id,
                        "from_chunk_id": left.id,
                        "to_chunk_id": right.id,
                        "ingestion_run_id": document.ingestion_run_id,
                        "chunking_policy_version": document.chunking_policy_version,
                    })

        async def transaction(tx):
            await _assert_no_conflicts(tx, _DOCUMENT_QUERY, doc_identity_rows)
            await _assert_no_conflicts(tx, _CHUNK_QUERY, chunk_rows)
            await _assert_has_chunk_identities(tx, has_chunk_rows)
            await _assert_next_chunk_identities(tx, next_rows)
            # MERGE plus this update takes the document write lock and returns
            # the authoritative current snapshot for an in-transaction ordering check.
            lock_result = await tx.run(_LOCK_DOCUMENT_QUERY, rows=doc_rows)
            locked_documents = await lock_result.data()
            lock_summary = await lock_result.consume()
            created = (lock_summary.counters.nodes_created
                       + lock_summary.counters.relationships_created)
            by_id = {_record_value(row, "id"): row for row in locked_documents}
            for document in documents:
                locked = by_id.get(document.source_sha256)
                if locked is None:
                    raise GraphIdentityConflict("Could not acquire document transaction lock")
                _reject_stale_run(
                    document,
                    _record_value(locked, "current_run_id"),
                    _record_value(locked, "current_created_at"),
                    _record_value(locked, "current_policy_version"),
                )
            created += await _run(tx, _WRITE_CHUNK_QUERY, chunk_rows)
            # Recheck identities under the held document lock before writing.
            await _assert_no_conflicts(tx, _DOCUMENT_QUERY, doc_identity_rows)
            await _assert_no_conflicts(tx, _CHUNK_QUERY, chunk_rows)
            await _assert_has_chunk_identities(tx, has_chunk_rows)
            await _assert_next_chunk_identities(tx, next_rows)
            created += await _run(tx, _UPDATE_DOCUMENT_CURRENT_QUERY, doc_rows)
            created += await _run(tx, _HAS_CHUNK_QUERY, has_chunk_rows)
            created += await _run(tx, _MARK_HISTORICAL_CHUNKS_QUERY, doc_rows)
            created += await _run(tx, _MARK_CURRENT_CHUNKS_QUERY, has_chunk_rows)
            created += await _run(tx, _NEXT_CHUNK_QUERY, next_rows)
            return created

        async with db_connection.get_session() as session:
            created = await session.execute_write(transaction)
        return {"documents": len(doc_rows), "chunks": len(chunk_rows), "has_chunk": len(has_chunk_rows),
                "next_chunk": len(next_rows), "created_records": created}
