"""Opt-in writes only to an explicitly named isolated Neo4j test database."""

import asyncio
import hashlib
import os
import uuid

import pytest

from src.graph.knowledge_extraction import proposal_to_batch
from src.graph.knowledge_writer import GraphIdentityConflict, persist_approved_batch
from tests.graph.test_knowledge_extraction import _attestations, _proposal
from tests.graph.test_graph_write_gate import _screened


pytestmark = pytest.mark.neo4j_integration


def _configured_database():
    name = os.getenv("NEO4J_TEST_DATABASE", "").strip()
    if not name:
        pytest.skip("set NEO4J_TEST_DATABASE to an isolated database with 'test' in its name")
    if "test" not in name.casefold():
        pytest.fail("Refusing Neo4j integration writes: test database name must contain 'test'")
    if name == os.getenv("NEO4J_DATABASE", "neo4j"):
        pytest.fail("Refusing Neo4j integration writes against NEO4J_DATABASE")
    values = [os.getenv(key) for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")]
    if any(not value for value in values):
        pytest.skip("set Neo4j credentials for the isolated test database")
    from src.infrastructure.neo4j_client import DbConnect
    return DbConnect(url=values[0], username=values[1], password=values[2], database=name)


def _batch(text, document_id, run_id, organization, application):
    document, screened = _screened(text, document_id=document_id)
    initial = _proposal(text)
    entities = (
        initial.entities[0].model_copy(update={"properties": {"name": organization}}),
        initial.entities[1].model_copy(update={"properties": {"name": application}}),
    )
    proposal = initial.model_copy(update={"entities": entities})
    batch = proposal_to_batch(
        proposal, document=document, screened=screened,
        support_attestations=_attestations(proposal, document),
        extraction_provider="integration-test", extraction_model="test-extractor",
    )
    return batch.model_copy(update={
        "ingestion_run_id": run_id,
        "extraction_provider": "integration-test",
        "extraction_model": "test-extractor",
    })


def test_semantic_writer_replay_cross_document_provenance_and_rollback():
    db = _configured_database()
    suffix = uuid.uuid4().hex
    organization = f"Test Court {suffix}"
    application = f"Test Filing {suffix}"
    text = f"The {organization} owns the {application}."
    doc_a = hashlib.sha256((text + "a").encode()).hexdigest()
    doc_b = hashlib.sha256((text + "b").encode()).hexdigest()
    run_a, run_b = f"run-a-{suffix}", f"run-b-{suffix}"
    batch_a = _batch(text, doc_a, run_a, organization, application)
    batch_b = _batch(text, doc_b, run_b, organization, application)
    node_ids = [str(node.properties["id"]) for node in batch_a.nodes]
    relation_ids = [str(edge.id) for edge in batch_a.relationships]

    async def run():
        try:
            await db.verify_connection()
            first = await persist_approved_batch(db, batch_a)
            replay = await persist_approved_batch(db, batch_a)
            second_document = await persist_approved_batch(db, batch_b)
            assert first == replay == second_document == {"nodes": 2, "relationships": 1}
            async with db.get_session() as session:
                result = await session.run(
                    "MATCH (n:ORGANIZATION {id:$id}) RETURN n.source_sha256s AS hashes, "
                    "n.ingestion_run_ids AS runs, n.evidence_references AS evidence, "
                    "n.confidence AS confidence, n.epistemic_mode AS epistemic",
                    id=node_ids[0],
                )
                record = await result.single()
                assert set(record["hashes"]) == {doc_a, doc_b}
                assert set(record["runs"]) == {run_a, run_b}
                assert len(record["evidence"]) == 2
                assert record["confidence"] >= .90
                assert record["epistemic"] == "VERBATIM_EXTRACTION"

                # Corrupt only this fixture to trigger a conflict after MERGE;
                # execute_write must roll back its attempted provenance append.
                await session.run(
                    "MATCH (n:ORGANIZATION {id:$id}) SET n.name=$name",
                    id=node_ids[0], name=organization + " changed",
                ).consume()
                with pytest.raises(GraphIdentityConflict):
                    await persist_approved_batch(db, batch_a)
                result = await session.run(
                    "MATCH (n:ORGANIZATION {id:$id}) RETURN n.name AS name, "
                    "n.source_sha256s AS hashes",
                    id=node_ids[0],
                )
                after = await result.single()
                assert after["name"] == organization + " changed"
                assert set(after["hashes"]) == {doc_a, doc_b}
        finally:
            try:
                async with db.get_session() as session:
                    await session.run(
                        "MATCH ()-[r:OWNS]-() WHERE r.id IN $ids DELETE r", ids=relation_ids
                    ).consume()
                    await session.run(
                        "MATCH (n) WHERE n.id IN $ids DETACH DELETE n", ids=node_ids
                    ).consume()
            finally:
                await db.close()

    asyncio.run(run())
