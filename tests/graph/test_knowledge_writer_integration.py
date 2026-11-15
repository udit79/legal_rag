"""Opt-in writes only to an explicitly named isolated Neo4j test database."""

import asyncio
import hashlib
import json
import os
import uuid

import pytest

from src.graph.knowledge_extraction import proposal_to_batch
from src.graph.knowledge_writer import GraphIdentityConflict, persist_approved_batch
from src.graph.assertion_contracts import (
    AssertionClassification,
    AssertionClassificationProposal,
)
from src.graph.semantic_schema import (
    SemanticSchemaPreflightError,
    initialize_semantic_schema,
    preflight_semantic_identities,
)
from src.graph.knowledge_writer import _batch_assertions, persist_assertion_classification
from tests.graph.test_knowledge_extraction import _attestations, _proposal
from tests.graph.test_graph_write_gate import _screened


pytestmark = pytest.mark.neo4j_integration


def _configured_database():
    from dotenv import load_dotenv

    load_dotenv(override=False)
    name = os.getenv("NEO4J_TEST_DATABASE", "").strip()
    if not name:
        pytest.skip("set NEO4J_TEST_DATABASE to an isolated database with 'test' in its name")
    if "test" not in name.casefold():
        pytest.fail("Refusing Neo4j integration writes: test database name must contain 'test'")
    configured_database = os.getenv("NEO4J_DATABASE", "neo4j").strip()
    if name == configured_database:
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
    doc_c = hashlib.sha256((text + "c").encode()).hexdigest()
    run_c = f"run-c-{suffix}"
    conflict_text = f"The {organization} owns the new application Review {suffix}."
    batch_c = _batch(conflict_text, doc_c, run_c, organization, f"Review {suffix}")
    rollback_sha = hashlib.sha256((text + "rollback").encode()).hexdigest()
    rollback_run = f"run-rollback-{suffix}"
    rollback_batch = _batch(text, rollback_sha, rollback_run, organization, application)
    all_batches = (batch_a, batch_b, batch_c)
    node_ids = sorted({str(node.properties["id"]) for batch in all_batches for node in batch.nodes})
    relation_ids = sorted({str(edge.id) for batch in all_batches for edge in batch.relationships})
    assertion_ids = sorted({
        assertion.id for batch in all_batches for assertion, _, _ in _batch_assertions(batch)
    })
    fixture_ids = sorted(set(node_ids) | set(assertion_ids))
    owned_ids: list[str] = []
    organization_id = next(
        str(node.properties["id"]) for node in batch_a.nodes if node.node_type == "ORGANIZATION"
    )
    application_id = next(
        str(node.properties["id"]) for node in batch_a.nodes if node.node_type == "APPLICATION"
    )

    async def run():
        try:
            await db.verify_connection()
            preflight = await preflight_semantic_identities(db)
            assert preflight["node_identity_issues"] == 0
            assert preflight["relationship_identity_issues"] == 0
            schema = await initialize_semantic_schema(db)
            assert schema["constraints_verified"] > 0
            duplicate_id = f"phase5-preflight-{suffix}"
            async with db.get_session() as session:
                await (await session.run(
                    "CREATE (:ORGANIZATION {id:$id, name:'preflight one'}), "
                    "(:APPLICATION {id:$id, name:'preflight two'})",
                    id=duplicate_id,
                )).consume()
            try:
                with pytest.raises(SemanticSchemaPreflightError):
                    await preflight_semantic_identities(db)
            finally:
                async with db.get_session() as session:
                    await (await session.run(
                        "MATCH (n) WHERE n.id=$id DETACH DELETE n", id=duplicate_id
                    )).consume()
            async with db.get_session() as session:
                existing = await session.run(
                    "MATCH (n) WHERE n.id IN $ids RETURN count(n) AS count", ids=fixture_ids
                )
                if (await existing.single())["count"]:
                    pytest.fail("Random integration fixture IDs unexpectedly already exist")
                owned_ids = fixture_ids
            concurrent_batches = [batch_a, batch_b] * 3
            concurrent = await asyncio.gather(*[
                persist_approved_batch(db, batch) for batch in concurrent_batches
            ])
            assert concurrent == [{"nodes": 2, "relationships": 1}] * 6
            first = concurrent[0]
            async with db.get_session() as session:
                result = await session.run(
                    "MATCH (n:ORGANIZATION {id:$id}) RETURN n.source_sha256s AS hashes, "
                    "n.ingestion_run_ids AS runs, n.evidence_references AS evidence",
                    id=organization_id,
                )
                concurrent_provenance = await result.single()
                assert set(concurrent_provenance["hashes"]) == {doc_a, doc_b}
                assert set(concurrent_provenance["runs"]) == {run_a, run_b}
                assert len(concurrent_provenance["evidence"]) == 2
            replay = await persist_approved_batch(db, batch_a)
            second_document = await persist_approved_batch(db, batch_b)
            conflict_document = await persist_approved_batch(db, batch_c)
            assert first == replay == second_document == conflict_document == {"nodes": 2, "relationships": 1}
            assertion_a = next(
                assertion for assertion, _, _ in _batch_assertions(batch_a)
                if assertion.predicate == "OWNS"
            )
            assertion_c = next(
                assertion for assertion, _, _ in _batch_assertions(batch_c)
                if assertion.predicate == "OWNS"
            )
            evidence_a = next(
                references for assertion, references, _ in _batch_assertions(batch_a)
                if assertion.predicate == "OWNS"
            )
            evidence_c = next(
                references for assertion, references, _ in _batch_assertions(batch_c)
                if assertion.predicate == "OWNS"
            )
            for classification in AssertionClassification:
                if classification is AssertionClassification.UNRESOLVED:
                    continue
                proposal = AssertionClassificationProposal(
                    source_assertion_id=assertion_a.id,
                    target_assertion_id=assertion_c.id,
                    classification=classification,
                    classifier="integration-test-proposal",
                    rationale="Persist a sourced classification proposal without legal adjudication.",
                    source_sha256s=(doc_a, doc_c),
                    ingestion_run_ids=(run_a, run_c),
                    evidence_references=tuple(json.dumps(
                        item, sort_keys=True, separators=(",", ":")
                    ) for item in (*evidence_a, *evidence_c)),
                )
                await asyncio.gather(*[
                    persist_assertion_classification(db, proposal) for _ in range(3)
                ])
            async with db.get_session() as session:
                result = await session.run(
                    "MATCH (n) WHERE n.id IN $ids RETURN count(n) AS count",
                    ids=node_ids,
                )
                assert (await result.single())["count"] == len(node_ids) == 3
                result = await session.run(
                    "MATCH ()-[r]->() WHERE r.id IN $ids "
                    "RETURN collect({id:r.id, source:startNode(r).id, "
                    "target:endNode(r).id, type:type(r)}) AS records",
                    ids=relation_ids,
                )
                relationship_records = (await result.single())["records"]
                assert len(relationship_records) == len(relation_ids) == 2, relationship_records
                assert {record["id"] for record in relationship_records} == set(relation_ids)
                assert all(record["type"] == "OWNS" for record in relationship_records)
                result = await session.run(
                    "MATCH (n:ORGANIZATION {id:$id}) RETURN n.source_sha256s AS hashes, "
                    "n.ingestion_run_ids AS runs, n.evidence_references AS evidence, "
                    "n.confidence AS confidence, n.epistemic_mode AS epistemic",
                    id=organization_id,
                )
                record = await result.single()
                assert set(record["hashes"]) == {doc_a, doc_b, doc_c}
                assert set(record["runs"]) == {run_a, run_b, run_c}
                assert len(record["evidence"]) == 3
                assert record["confidence"] >= .90
                assert record["epistemic"] == "VERBATIM_EXTRACTION"

                result = await session.run(
                    "MATCH (a:KNOWLEDGE_ASSERTION) WHERE a.id IN $ids "
                    "RETURN collect({id:a.id, hashes:a.source_sha256s, "
                    "runs:a.ingestion_run_ids, evidence:a.evidence_references, "
                    "attestations:a.support_attestations}) AS records",
                    ids=assertion_ids,
                )
                assertion_record = await result.single()
                assertion_records = assertion_record["records"]
                assert len(assertion_records) == len(assertion_ids)
                assert {value for item in assertion_records for value in item["hashes"]} == {doc_a, doc_b, doc_c}
                assert {value for item in assertion_records for value in item["runs"]} == {run_a, run_b, run_c}
                for item in assertion_records:
                    source_texts = {doc_a: text, doc_b: text, doc_c: conflict_text}
                    references = [json.loads(value) for value in item["evidence"]]
                    assert {value["source_sha256"] for value in references} == set(item["hashes"])
                    assert {value["ingestion_run_id"] for value in references} == set(item["runs"])
                    assert all(
                        value["quote"] in source_texts[value["source_sha256"]]
                        and value["page_number"] == 1
                        and value["block_id"]
                        for value in references
                    )
                    attestations = [json.loads(value) for value in item["attestations"]]
                    assert attestations
                    assert all(
                        value["supported"] and value["directly_stated"]
                        and value["model"] == "independent-test-verifier"
                        for value in attestations
                    )

                result = await session.run(
                    "MATCH (a:KNOWLEDGE_ASSERTION)-[r:ASSERTION_CLASSIFICATION]->(b:KNOWLEDGE_ASSERTION) "
                    "WHERE a.id IN $ids OR b.id IN $ids "
                    "RETURN r.classification AS classification, r.adjudication_status AS status, "
                    "r.source_sha256s AS hashes, r.ingestion_run_ids AS runs, "
                    "r.evidence_references AS evidence, r.rationale AS rationale",
                    ids=assertion_ids,
                )
                classifications = await result.data()
                assert {item["classification"] for item in classifications} == {
                    value.value for value in AssertionClassification
                }
                assert all(item["status"] == "NOT_ADJUDICATED" for item in classifications)
                unresolved = next(
                    item for item in classifications
                    if item["classification"] == AssertionClassification.UNRESOLVED.value
                )
                assert unresolved["classification"] == AssertionClassification.UNRESOLVED.value
                assert set(unresolved["hashes"]) == {doc_a, doc_b, doc_c}
                assert set(unresolved["runs"]) == {run_a, run_b, run_c}
                assert len(unresolved["evidence"]) == 3
                assert "no truth adjudication" in unresolved["rationale"]

                # Corrupt the second node so the first node's new provenance append
                # must be rolled back when the later identity check fails.
                await (await session.run(
                    "MATCH (n:APPLICATION {id:$id}) SET n.name=$name",
                    id=application_id,
                    name=application + " changed",
                )).consume()
                with pytest.raises(GraphIdentityConflict):
                    await persist_approved_batch(db, rollback_batch)
                result = await session.run(
                    "MATCH (n:ORGANIZATION {id:$id}) RETURN n.source_sha256s AS hashes, "
                    "n.ingestion_run_ids AS runs, n.evidence_references AS evidence, "
                    "n.extraction_assertions AS assertions",
                    id=organization_id,
                )
                after = await result.single()
                assert set(after["hashes"]) == {doc_a, doc_b, doc_c}
                assert set(after["runs"]) == {run_a, run_b, run_c}
                assert len(after["evidence"]) == 3
                assert len(after["assertions"]) == 3
                result = await session.run(
                    "MATCH (n:APPLICATION {id:$id}) RETURN n.name AS name",
                    id=application_id,
                )
                assert (await result.single())["name"] == application + " changed"
        finally:
            try:
                async with db.get_session() as session:
                    if owned_ids:
                        await (await session.run(
                                "MATCH (a)-[r]->(b) WHERE a.id IN $ids OR b.id IN $ids DELETE r",
                                ids=owned_ids,
                        )).consume()
                        await (await session.run(
                            "MATCH (n) WHERE n.id IN $ids DETACH DELETE n", ids=owned_ids
                        )).consume()
                        remaining_nodes = await session.run(
                            "MATCH (n) WHERE n.id IN $ids RETURN count(n) AS count",
                            ids=owned_ids,
                        )
                        if (await remaining_nodes.single())["count"] != 0:
                            raise AssertionError("Neo4j integration fixture nodes were not cleaned up")
                        remaining_relationships = await session.run(
                            "MATCH (a)-[r]->(b) WHERE a.id IN $ids OR b.id IN $ids "
                            "RETURN count(r) AS count",
                            ids=owned_ids,
                        )
                        if (await remaining_relationships.single())["count"] != 0:
                            raise AssertionError("Neo4j integration fixture relationships were not cleaned up")
            finally:
                await db.close()

    asyncio.run(run())
