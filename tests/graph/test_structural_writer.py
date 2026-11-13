from types import SimpleNamespace
import asyncio
import hashlib
import os
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.graph.corpus_reader import CorpusChunk, CorpusDocument
from src.graph.structural_writer import (
    CURRENT_CHUNK_RETRIEVAL_QUERY,
    CURRENT_NEXT_CHUNK_QUERY,
    GraphIdentityConflict,
    StructuralGraphWriter,
    validate_structural_document,
)


def _doc(indices=(0, 1), decisions=None, run_id="run", policy="policy", created_at=None):
    decisions = decisions or ["ADMITTED"] * len(indices)
    chunks = tuple(CorpusChunk(
        id=hashlib.sha256(f"{'b' * 64}\0{policy}\0{i}".encode()).hexdigest(), chunk_index=position,
        original_chunk_indices=(i,), raw_text=f"raw {i}",
        retrieval_text=f"retrieval {i}", chunk_sha256="a" * 64,
        sources=[{"page_number": 1, "block_id": f"page-1-block-{i}"}],
        retrieval_text_edits=[], admission_decisions=[decisions[position]],
    ) for position, i in enumerate(indices))
    return CorpusDocument("b" * 64, "d" * 64, 1, run_id,
                          created_at or datetime(2026, 11, 12, tzinfo=timezone.utc), policy,
                          "screening-1", "c" * 64, chunks)


class _Result:
    def __init__(self, data=(), created=0):
        self._data = list(data)
        self._created = created

    async def data(self):
        return self._data

    async def consume(self):
        return SimpleNamespace(counters=SimpleNamespace(nodes_created=self._created,
                                                         relationships_created=self._created))


class _Tx:
    def __init__(self, conflict=False, additional_properties=False,
                 relationship_error=None, next_relationship_error=None,
                 current_run_id=None, current_created_at=None,
                 current_policy_version=None):
        self.conflict = conflict
        self.additional_properties = additional_properties
        self.relationship_error = relationship_error
        self.next_relationship_error = next_relationship_error
        self.current_run_id = current_run_id
        self.current_created_at = current_created_at
        self.current_policy_version = current_policy_version
        self.calls = []
        self.write_count = 0

    async def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if "RETURN d.id AS id" in query and "current_created_at" in query:
            rows = kwargs["rows"]
            return _Result([{"id": row["id"], "current_run_id": self.current_run_id,
                             "current_created_at": self.current_created_at,
                             "current_policy_version": self.current_policy_version}
                            for row in rows])
        if "SET d.current_chunking_policy_version" in query:
            row = kwargs["rows"][0]
            self.current_run_id = row["current_ingestion_run_id"]
            self.current_created_at = row["current_ingestion_created_at"]
            self.current_policy_version = row["current_chunking_policy_version"]
            self.write_count += 1
            return _Result(created=0 if self.write_count > 4 else 1)
        if "size(matches) AS match_count" in query:
            rows = kwargs["rows"]
            is_next = "NEXT_CHUNK" in query
            error = self.next_relationship_error if is_next else self.relationship_error
            if error == "duplicate":
                return _Result([{"id": row["id"], "match_count": 2, "matches": []}
                                for row in rows])
            if error == "wrong_endpoint":
                if is_next:
                    return _Result([{
                        "id": row["id"], "match_count": 1,
                        "matches": [{"properties": {key: row[key] for key in (
                            "id", "source_sha256", "from_chunk_id", "to_chunk_id",
                            "ingestion_run_id", "chunking_policy_version")},
                            "source_id": "wrong-chunk", "target_id": row["to_id"]}],
                    } for row in rows])
                return _Result([{
                    "id": row["id"], "match_count": 1,
                    "matches": [{"properties": {"id": row["id"],
                                                  "source_sha256": row["source_sha256"],
                                                  "chunk_id": row["chunk_id"]},
                                 "source_id": "wrong-document", "target_id": row["chunk_id"]}],
                } for row in rows])
            if error == "wrong_property":
                if is_next:
                    return _Result([{
                        "id": row["id"], "match_count": 1,
                        "matches": [{"properties": {**{key: row[key] for key in (
                            "id", "source_sha256", "from_chunk_id", "to_chunk_id",
                            "ingestion_run_id", "chunking_policy_version")},
                            "source_sha256": "wrong-source"},
                            "source_id": row["from_id"], "target_id": row["to_id"]}],
                    } for row in rows])
                return _Result([{
                    "id": row["id"], "match_count": 1,
                    "matches": [{"properties": {"id": row["id"],
                                                  "source_sha256": "wrong-source",
                                                  "chunk_id": row["chunk_id"]},
                                 "source_id": row["source_sha256"],
                                 "target_id": row["chunk_id"]}],
                } for row in rows])
            if self.additional_properties and not is_next:
                return _Result([{
                    "id": row["id"], "match_count": 1,
                    "matches": [{"properties": {"id": row["id"],
                                                  "source_sha256": row["source_sha256"],
                                                  "chunk_id": row["chunk_id"],
                                                  "ingestion_run_history": ["history"],
                                                  "semantic_property": "keep"},
                                 "source_id": row["source_sha256"],
                                 "target_id": row["chunk_id"]}],
                } for row in rows])
            return _Result([{"id": row["id"], "match_count": 0, "matches": []}
                            for row in rows])
        if "properties(d)" in query or "properties(c)" in query:
            rows = kwargs["rows"]
            if self.conflict and "properties(c)" in query:
                return _Result([{"id": rows[0]["id"], "existing": {**rows[0], "raw_text": "changed"}}])
            return _Result([{
                "id": row["id"],
                "existing": ({**row, "embedding": [0.1, 0.2], "semantic_label": "added"}
                             if self.additional_properties else None),
            } for row in rows])
        self.write_count += 1
        return _Result(created=0 if self.write_count > 4 else 1)


class _Session:
    def __init__(self, tx):
        self.tx = tx

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def execute_write(self, callback):
        return await callback(self.tx)


class _Db:
    def __init__(self, tx):
        self.tx = tx

    def get_session(self):
        return _Session(self.tx)


def test_writer_only_links_source_proven_adjacency_and_is_idempotent():
    async def run():
        tx = _Tx()
        writer = StructuralGraphWriter()
        counts = await writer.write(_Db(tx), (_doc(),))
        assert counts["chunks"] == 2
        document_write = next(kwargs for query, kwargs in tx.calls
                              if "MERGE (d:DOCUMENT" in query)
        assert document_write["rows"][0]["canonical_document_digest"] == "d" * 64
        assert document_write["rows"][0]["source_page_count"] == 1
        next_calls = [kwargs for query, kwargs in tx.calls if "NEXT_CHUNK" in query]
        assert len(next_calls[0]["rows"]) == 1
        edge = next_calls[0]["rows"][0]
        assert edge["id"] == hashlib.sha256(
            f"{'b' * 64}\0{_doc().chunks[0].id}\0{_doc().chunks[1].id}\0run".encode()
        ).hexdigest()
        assert edge["ingestion_run_id"] == "run"
        assert edge["chunking_policy_version"] == "policy"
        assert edge["from_chunk_id"] == edge["from_id"]
        assert edge["to_chunk_id"] == edge["to_id"]
        identity_checks = [query for query, _ in tx.calls
                           if "size(matches) AS match_count" in query
                           and "NEXT_CHUNK" in query]
        # Each edge is checked before writes and again after the document lock.
        assert len(identity_checks) == 2
        assert "current_chunking_policy_version" in document_write["rows"][0]
        rerun = await writer.write(_Db(tx), (_doc(),))
        assert rerun["created_records"] == 0
    asyncio.run(run())


def test_writer_does_not_bridge_original_chunk_gaps():
    async def run():
        tx = _Tx()
        await StructuralGraphWriter().write(_Db(tx), (_doc((0, 2)),))
        next_calls = [kwargs for query, kwargs in tx.calls if "NEXT_CHUNK" in query]
        assert next_calls == []
    asyncio.run(run())


def test_quarantined_chunks_cannot_enter_graph():
    with pytest.raises(ValueError, match="Quarantined or rejected"):
        validate_structural_document(_doc(decisions=["QUARANTINED", "ADMITTED"]))


def test_duplicate_deterministic_chunk_identity_is_rejected():
    with pytest.raises(GraphIdentityConflict, match="same deterministic source identity"):
        validate_structural_document(_doc((0, 0)))


def test_writer_rejects_empty_structural_output():
    empty = CorpusDocument("b" * 64, "d" * 64, 1, "run",
                           datetime(2026, 11, 12, tzinfo=timezone.utc), "policy",
                           "screening-1", "c" * 64, ())
    with pytest.raises(ValueError, match="at least one admitted chunk"):
        validate_structural_document(empty)


def test_writer_rejects_conflicting_content_for_existing_identity():
    async def run():
        with pytest.raises(GraphIdentityConflict):
            await StructuralGraphWriter().write(_Db(_Tx(conflict=True)), (_doc(),))
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["wrong_endpoint", "wrong_property", "duplicate"])
def test_writer_rejects_wrong_or_duplicate_has_chunk_identities(failure):
    async def run():
        with pytest.raises(GraphIdentityConflict, match="HAS_CHUNK identity"):
            await StructuralGraphWriter().write(
                _Db(_Tx(relationship_error=failure)), (_doc(),)
            )
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["wrong_endpoint", "wrong_property", "duplicate"])
def test_writer_rejects_wrong_or_duplicate_next_chunk_identities(failure):
    async def run():
        with pytest.raises(GraphIdentityConflict, match="NEXT_CHUNK identity"):
            await StructuralGraphWriter().write(
                _Db(_Tx(next_relationship_error=failure)), (_doc(),)
            )
    asyncio.run(run())


def test_repeated_ingestion_uses_one_stable_has_chunk_with_separate_run_history():
    async def run():
        tx = _Tx()
        writer = StructuralGraphWriter()
        await writer.write(_Db(tx), (_doc(run_id="run-one"),))
        await writer.write(_Db(tx), (_doc(
            run_id="run-two", created_at=datetime(2026, 11, 13, tzinfo=timezone.utc)),))
        writes = [kwargs["rows"] for query, kwargs in tx.calls
                  if "MERGE (d)-[r:HAS_CHUNK" in query]
        first_rows, second_rows = writes
        assert [row["id"] for row in first_rows] == [row["id"] for row in second_rows]
        assert all("run-one" in row["run_history_entry"] for row in first_rows)
        assert all("run-two" in row["run_history_entry"] for row in second_rows)
        assert "ON MATCH SET r.ingestion_run_history" in next(
            query for query, _ in tx.calls if "MERGE (d)-[r:HAS_CHUNK" in query
        )
    asyncio.run(run())


def test_next_chunk_keeps_changed_adjacency_separate_by_ingestion_run():
    async def run():
        tx = _Tx()
        writer = StructuralGraphWriter()
        previous = _doc(indices=(0, 1, 2), run_id="previous-run")
        current = _doc(indices=(0, 2, 3), run_id="current-run",
                       created_at=datetime(2026, 11, 13, tzinfo=timezone.utc))
        await writer.write(_Db(tx), (previous,))
        await writer.write(_Db(tx), (current,))
        next_writes = [kwargs["rows"] for query, kwargs in tx.calls
                       if "MERGE (a)-[r:NEXT_CHUNK" in query]
        assert [(row["from_id"], row["to_id"]) for row in next_writes[0]] == [
            (previous.chunks[0].id, previous.chunks[1].id),
            (previous.chunks[1].id, previous.chunks[2].id),
        ]
        assert [(row["from_id"], row["to_id"]) for row in next_writes[1]] == [
            (current.chunks[1].id, current.chunks[2].id),
        ]
        assert all(row["ingestion_run_id"] == "previous-run" for row in next_writes[0])
        assert next_writes[1][0]["ingestion_run_id"] == "current-run"
        all_ids = [row["id"] for rows in next_writes for row in rows]
        assert len(all_ids) == len(set(all_ids))
        assert "r.ingestion_run_id = d.current_ingestion_run_id" in CURRENT_NEXT_CHUNK_QUERY
        assert "r.chunking_policy_version = d.current_chunking_policy_version" in CURRENT_NEXT_CHUNK_QUERY
        assert "r.source_sha256 = d.source_sha256" in CURRENT_NEXT_CHUNK_QUERY
    asyncio.run(run())


def test_unchanged_adjacency_is_retained_as_run_specific_history():
    async def run():
        tx = _Tx()
        writer = StructuralGraphWriter()
        first = _doc(run_id="run-one")
        second = _doc(run_id="run-two",
                      created_at=datetime(2026, 11, 13, tzinfo=timezone.utc))
        await writer.write(_Db(tx), (first,))
        await writer.write(_Db(tx), (second,))
        next_writes = [kwargs["rows"] for query, kwargs in tx.calls
                       if "MERGE (a)-[r:NEXT_CHUNK" in query]
        first_edge, second_edge = next_writes[0][0], next_writes[1][0]
        assert (first_edge["from_id"], first_edge["to_id"]) == (
            second_edge["from_id"], second_edge["to_id"]
        )
        assert first_edge["id"] != second_edge["id"]
        assert first_edge["ingestion_run_id"] == "run-one"
        assert second_edge["ingestion_run_id"] == "run-two"
        assert first_edge["chunking_policy_version"] == second_edge["chunking_policy_version"]
    asyncio.run(run())


def test_existing_nodes_with_additional_properties_do_not_conflict():
    async def run():
        await StructuralGraphWriter().write(
            _Db(_Tx(additional_properties=True)), (_doc(),)
        )
    asyncio.run(run())


def test_changed_chunking_policy_marks_old_chunks_historical():
    async def run():
        tx = _Tx()
        writer = StructuralGraphWriter()
        old = _doc(run_id="old-run", policy="policy-v1")
        current = _doc(run_id="new-run", policy="policy-v2",
                       created_at=datetime(2026, 11, 13, tzinfo=timezone.utc))
        await writer.write(_Db(tx), (old,))
        await writer.write(_Db(tx), (current,))
        old_ids = {chunk.id for chunk in old.chunks}
        new_ids = {chunk.id for chunk in current.chunks}
        assert old_ids.isdisjoint(new_ids)
        historical_query = next(query for query, _ in tx.calls
                                if "SET c.is_current = false" in query)
        current_query = next(query for query, _ in tx.calls
                             if "SET c.is_current = true" in query)
        assert "HAS_CHUNK" in historical_query
        assert "HAS_CHUNK" in current_query
        assert "c.is_current = true" in CURRENT_CHUNK_RETRIEVAL_QUERY
        assert "CYPHER 25" in CURRENT_CHUNK_RETRIEVAL_QUERY
        assert "SEARCH c IN (" in CURRENT_CHUNK_RETRIEVAL_QUERY
        assert "FULLTEXT INDEX chunk_retrieval_text_fulltext" in CURRENT_CHUNK_RETRIEVAL_QUERY
        assert "LIMIT $candidate_count" in CURRENT_CHUNK_RETRIEVAL_QUERY
        assert "LIMIT $limit" in CURRENT_CHUNK_RETRIEVAL_QUERY
        assert "queryNodes" not in CURRENT_CHUNK_RETRIEVAL_QUERY
        assert "c.chunking_policy_version = d.current_chunking_policy_version" in CURRENT_CHUNK_RETRIEVAL_QUERY
        assert "a.is_current = true" in CURRENT_NEXT_CHUNK_QUERY
        assert "b.is_current = true" in CURRENT_NEXT_CHUNK_QUERY
        assert "a.chunking_policy_version = d.current_chunking_policy_version" in CURRENT_NEXT_CHUNK_QUERY
        assert "b.chunking_policy_version = d.current_chunking_policy_version" in CURRENT_NEXT_CHUNK_QUERY
        doc_rows = [kwargs["rows"][0] for query, kwargs in tx.calls
                    if "MERGE (d:DOCUMENT" in query]
        assert doc_rows[-1]["current_chunking_policy_version"] == "policy-v2"
        next_writes = [kwargs["rows"] for query, kwargs in tx.calls
                       if "MERGE (a)-[r:NEXT_CHUNK" in query]
        assert next_writes[0][0]["chunking_policy_version"] == "policy-v1"
        assert next_writes[1][0]["chunking_policy_version"] == "policy-v2"
        assert next_writes[0][0]["id"] != next_writes[1][0]["id"]
        current_marks = [kwargs["rows"] for query, kwargs in tx.calls
                         if "SET c.is_current = true" in query]
        assert [row["chunk_id"] for row in current_marks[0]] == [chunk.id for chunk in old.chunks]
        assert [row["chunk_id"] for row in current_marks[1]] == [chunk.id for chunk in current.chunks]
    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("NEO4J_INTEGRATION") != "1",
                    reason="set NEO4J_INTEGRATION=1 for configured Neo4j writer integration")
def test_configured_neo4j_writer_integration():
    async def run():
        from src.graph.schema_bootstrap import initialize_graph_schema
        from src.infrastructure.neo4j_client import create_db_connect

        run_token = uuid.uuid4().hex
        source_sha = hashlib.sha256(run_token.encode()).hexdigest()
        raw = "Opt-in structural graph writer integration fixture."
        chunk_sha = hashlib.sha256(raw.encode()).hexdigest()
        chunk_id = hashlib.sha256(f"{source_sha}\0policy\0{0}".encode()).hexdigest()
        document = CorpusDocument(
            source_sha, hashlib.sha256(b"integration document").hexdigest(), 1,
            f"integration-{run_token}", datetime.now(timezone.utc),
            "policy", "screening-1", "c" * 64,
            (CorpusChunk(chunk_id, 0, (0,), raw, raw, chunk_sha,
                         [{"page_number": 1, "block_id": "page-1-block-0"}], [],
                         ["ADMITTED"]),),
        )
        db = create_db_connect()
        try:
            await db.verify_connection()
            await initialize_graph_schema(db)
            writer = StructuralGraphWriter()
            await writer.write(db, (document,))
            repeated = await writer.write(db, (document,))
            assert repeated["created_records"] == 0
        finally:
            await db.close()
    asyncio.run(run())


def test_stale_reordered_ingestion_cannot_replace_newer_current_run():
    async def run():
        tx = _Tx()
        writer = StructuralGraphWriter()
        newer = _doc(run_id="newer", created_at=datetime(2026, 11, 12, 12, tzinfo=timezone.utc))
        older = _doc(run_id="older", created_at=datetime(2026, 11, 12, 11, tzinfo=timezone.utc))
        await writer.write(_Db(tx), (newer,))
        with pytest.raises(GraphIdentityConflict, match="Stale ingestion run older"):
            await writer.write(_Db(tx), (older,))
        assert tx.current_run_id == "newer"
        assert tx.current_created_at == newer.ingestion_created_at.isoformat()
    asyncio.run(run())


def test_newer_run_replaces_older_current_run_under_document_lock():
    async def run():
        tx = _Tx()
        writer = StructuralGraphWriter()
        older = _doc(run_id="older", created_at=datetime(2026, 11, 12, 11, tzinfo=timezone.utc))
        newer = _doc(run_id="newer", created_at=datetime(2026, 11, 12, 12, tzinfo=timezone.utc))
        await writer.write(_Db(tx), (older,))
        await writer.write(_Db(tx), (newer,))
        assert tx.current_run_id == "newer"
        assert tx.current_created_at == newer.ingestion_created_at.isoformat()
        lock_positions = [i for i, (query, _) in enumerate(tx.calls)
                          if "current_created_at" in query]
        update_positions = [i for i, (query, _) in enumerate(tx.calls)
                            if "SET d.current_chunking_policy_version" in query]
        assert len(lock_positions) == len(update_positions) == 2
        assert all(lock < update for lock, update in zip(lock_positions, update_positions))
    asyncio.run(run())


def test_repeated_same_ingestion_run_is_idempotently_accepted():
    async def run():
        tx = _Tx()
        writer = StructuralGraphWriter()
        document = _doc(run_id="same-run")
        await writer.write(_Db(tx), (document,))
        assert (await writer.write(_Db(tx), (document,))) ["created_records"] == 0
    asyncio.run(run())


def test_equal_timestamps_with_different_run_ids_are_ambiguous_in_utc():
    async def run():
        tx = _Tx()
        writer = StructuralGraphWriter()
        first = _doc(run_id="run-a", created_at=datetime(2026, 11, 12, 12, tzinfo=timezone.utc))
        later_id = _doc(run_id="run-b", created_at=datetime(
            2026, 11, 12, 7, tzinfo=timezone(timedelta(hours=-5))))
        await writer.write(_Db(tx), (first,))
        with pytest.raises(GraphIdentityConflict, match="equal timestamps"):
            await writer.write(_Db(tx), (later_id,))
        assert tx.current_run_id == "run-a"
        assert tx.current_created_at == "2026-11-12T12:00:00+00:00"
    asyncio.run(run())


def test_structural_validation_rejects_naive_ingestion_timestamp():
    with pytest.raises(ValueError, match="timezone-aware datetime"):
        validate_structural_document(replace(
            _doc(), ingestion_created_at=datetime(2026, 11, 12, 12)
        ))
