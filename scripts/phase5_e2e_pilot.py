"""Run a small real-model extraction through the isolated Neo4j graph writer.

Requires NEO4J_TEST_DATABASE (its name must include ``test``), Neo4j
credentials, OPENAI_API_KEY, and separate GRAPH_EXTRACTION_MODEL and
GRAPH_EVIDENCE_VERIFIER_MODEL values. All graph fixtures use random identities
and are removed in a finally block.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import string
import sys
import tempfile
import uuid
from pathlib import Path

import pymupdf
from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env", override=False)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.graph_write_gate import validate_graph_write_batch
from src.graph.knowledge_extraction import (
    OpenAIEvidenceVerifier,
    OpenAIStructuredExtractor,
    StructuredExtractionConfig,
    extract_validated_batch,
)
from src.graph.knowledge_writer import _batch_assertions, persist_approved_batch
from src.graph.semantic_schema import initialize_semantic_schema, preflight_semantic_identities
from src.ingestion.chunking import ChunkingConfig, chunk_document
from src.ingestion.parser_contracts import CanonicalDocument
from src.ingestion.parsers import FastTextParser
from src.infrastructure.neo4j_client import DbConnect
from src.security.ingestion_screening.contracts import DocumentContext
from src.security.ingestion_screening.enums import TrustTier
from src.security.ingestion_screening.screening_service import ScreeningService


async def _build_batch(text: str, run_id: str, temp_dir: Path, extractor, verifier):
    pdf_path = temp_dir / f"{run_id}.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    pdf.set_metadata({"title": run_id})
    pdf.save(pdf_path)
    pdf.close()

    source_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    parsed_page = FastTextParser().parse_page(pdf_path, 1)
    canonical = CanonicalDocument(
        page_count=1,
        document_disposition="PROCESS",
        pages=(parsed_page,),
    )
    canonical_digest = hashlib.sha256(
        json.dumps(
            canonical.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    chunking = chunk_document(canonical, source_sha256, ChunkingConfig())
    context = DocumentContext(
        source_id=f"phase5-pilot-{run_id}",
        ingestion_run_id=run_id,
        source_sha256=source_sha256,
        canonical_document_digest=canonical_digest,
        chunking_policy_version=chunking.document.policy_version,
        source_page_count=1,
        trust_tier=TrustTier.T3_UNAUTHENTICATED_EXTERNAL,
    )
    screened = await ScreeningService().screen_batch(
        chunking.document.chunks, context, document_id=source_sha256
    )
    admitted = {"ADMITTED", "ADMITTED_FLAGGED"}
    if not screened.report.chunk_results or any(
        result.decision.value not in admitted
        for result in screened.report.chunk_results
    ):
        raise RuntimeError("The synthetic pilot source did not pass screening")

    batch = await extract_validated_batch(
        extractor,
        verifier,
        document=chunking.document,
        screened=screened,
        ingestion_run_id=run_id,
        max_input_chars=4_000,
        max_output_tokens=2_500,
    )
    validate_graph_write_batch(batch)
    if len(batch.nodes) < 2 or not batch.relationships:
        raise RuntimeError(
            f"Expected at least two entities and one relation; got "
            f"{len(batch.nodes)} nodes and {len(batch.relationships)} relations"
        )
    return batch, source_sha256


class ReplayExtractor:
    """Reuse one real extraction for a byte-distinct, text-identical replay."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.provider = delegate.provider
        self.model = delegate.model
        self.response = None

    async def extract(self, *, payload, max_output_tokens):
        if self.response is None:
            self.response = await self.delegate.extract(
                payload=payload, max_output_tokens=max_output_tokens
            )
        return self.response


async def _assert_ids_available(db, node_ids: set[str], relationship_ids: set[str]) -> None:
    async with db.get_session() as session:
        if node_ids:
            result = await session.run(
                "MATCH (n) WHERE n.id IN $ids RETURN count(n) AS count",
                ids=list(node_ids),
            )
            if (await result.single())["count"]:
                raise RuntimeError("A generated pilot node identity already exists; refusing to overwrite it")
        if relationship_ids:
            result = await session.run(
                "MATCH ()-[r]->() WHERE r.id IN $ids RETURN count(r) AS count",
                ids=list(relationship_ids),
            )
            if (await result.single())["count"]:
                raise RuntimeError("A generated pilot relationship identity already exists")


async def run_pilot() -> dict[str, object]:
    database = os.getenv("NEO4J_TEST_DATABASE", "").strip()
    configured_database = os.getenv("NEO4J_DATABASE", "neo4j").strip()
    if not database or "test" not in database.casefold():
        raise ValueError("Set NEO4J_TEST_DATABASE to an isolated database containing 'test'")
    if database == configured_database:
        raise ValueError("Refusing to write the configured/default Neo4j database")

    config = StructuredExtractionConfig.from_env()
    extractor_client = OpenAIStructuredExtractor(config)
    extractor = ReplayExtractor(extractor_client)
    verifier = OpenAIEvidenceVerifier(config)
    db = DbConnect(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=database,
    )
    node_ids: set[str] = set()
    assertion_ids: set[str] = set()
    relationship_ids: set[str] = set()
    pilot_token = "".join(secrets.choice(string.ascii_uppercase) for _ in range(10))
    first_name = f"{pilot_token} Authority"
    application_name = f"{pilot_token} Registry"
    texts = (f"{first_name} owns software application {application_name}.",) * 2
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not configured")
        if extractor.model == verifier.model:
            raise ValueError("Extraction and evidence verifier models must differ")
        await db.verify_connection()
        preflight = await preflight_semantic_identities(db)
        semantic_schema = await initialize_semantic_schema(db)

        with tempfile.TemporaryDirectory(prefix="phase5-pilot-", dir=".") as temp:
            temp_dir = Path(temp)
            run_ids = tuple(f"phase5-pilot-{uuid.uuid4().hex}" for _ in texts)
            first, source_a = await _build_batch(
                texts[0], run_ids[0], temp_dir, extractor, verifier
            )
            first_node_ids = {str(node.properties["id"]) for node in first.nodes}
            first_relationship_ids = {str(edge.id) for edge in first.relationships}
            first_assertion_ids = {
                assertion.id for assertion, _, _ in _batch_assertions(first)
            }
            await _assert_ids_available(
                db, first_node_ids | first_assertion_ids, first_relationship_ids
            )
            node_ids.update(first_node_ids)
            assertion_ids.update(first_assertion_ids)
            relationship_ids.update(first_relationship_ids)
            first_write = await persist_approved_batch(db, first)
            replay = await persist_approved_batch(db, first)

            second, source_b = await _build_batch(
                texts[1], run_ids[1], temp_dir, extractor, verifier
            )
            if source_a == source_b:
                raise RuntimeError("Pilot source documents must have distinct hashes")
            second_node_ids = {str(node.properties["id"]) for node in second.nodes}
            second_relationship_ids = {str(edge.id) for edge in second.relationships}
            second_assertion_ids = {
                assertion.id for assertion, _, _ in _batch_assertions(second)
            }
            await _assert_ids_available(
                db,
                (second_node_ids | second_assertion_ids) - node_ids - assertion_ids,
                second_relationship_ids - relationship_ids,
            )
            node_ids.update(second_node_ids)
            assertion_ids.update(second_assertion_ids)
            relationship_ids.update(second_relationship_ids)
            second_write = await persist_approved_batch(db, second)
            shared_ids = {
                str(node.properties["id"]) for node in first.nodes
            } & {str(node.properties["id"]) for node in second.nodes}
            if not shared_ids:
                raise RuntimeError("Second document did not reuse a semantic entity ID")

            async with db.get_session() as session:
                result = await session.run(
                    "MATCH (n) WHERE n.id IN $ids "
                    "RETURN n.id AS id, n.source_sha256s AS hashes, "
                    "n.ingestion_run_ids AS runs, n.evidence_references AS evidence, "
                    "n.extraction_assertions AS assertions, n.confidence AS confidence, "
                    "n.epistemic_mode AS epistemic",
                    ids=list(shared_ids),
                )
                persisted_nodes = await result.data()
                if len(persisted_nodes) != len(shared_ids):
                    raise RuntimeError("Shared semantic nodes are missing or duplicated")
                for node in persisted_nodes:
                    if not {source_a, source_b}.issubset(set(node["hashes"])):
                        raise RuntimeError("Cross-document source hashes did not accumulate")
                    if not set(run_ids).issubset(set(node["runs"])):
                        raise RuntimeError("Cross-document ingestion runs did not accumulate")
                    if node["confidence"] < 0.90 or node["epistemic"] != "VERBATIM_EXTRACTION":
                        raise RuntimeError("Persisted node extraction metadata is invalid")
                    evidence = [json.loads(item) for item in node["evidence"]]
                    if len(evidence) != 2:
                        raise RuntimeError("Replay duplicated or lost evidence references")
                    for item in evidence:
                        source_text = texts[0] if item["source_sha256"] == source_a else texts[1]
                        if item["quote"] not in source_text or item["ingestion_run_id"] not in run_ids:
                            raise RuntimeError("Persisted quote or run does not match source text")
                    assertions = [json.loads(item) for item in node["assertions"]]
                    relevant_assertions = [
                        item for item in assertions
                        if any(ref["source_sha256"] in {source_a, source_b}
                               for ref in item["evidence_references"])
                    ]
                    if len(relevant_assertions) < 2 or any(
                        item["support_attestation"]["model"] != verifier.model
                        for item in relevant_assertions
                    ):
                        raise RuntimeError("Independent verifier metadata did not persist")

                result = await session.run(
                    "MATCH (a:KNOWLEDGE_ASSERTION) WHERE a.id IN $ids "
                    "RETURN a.id AS id, a.source_sha256s AS hashes, "
                    "a.ingestion_run_ids AS runs, a.evidence_references AS evidence, "
                    "a.support_attestations AS attestations",
                    ids=list(assertion_ids),
                )
                persisted_assertions = await result.data()
                if len(persisted_assertions) != len(assertion_ids):
                    raise RuntimeError("First-class assertion nodes are missing or duplicated")
                for assertion in persisted_assertions:
                    if not {source_a, source_b}.issubset(set(assertion["hashes"])):
                        raise RuntimeError("Assertion source hashes did not accumulate")
                    if not set(run_ids).issubset(set(assertion["runs"])):
                        raise RuntimeError("Assertion ingestion runs did not accumulate")
                    evidence = [json.loads(item) for item in assertion["evidence"]]
                    if len(evidence) != 2 or any(
                        item["source_sha256"] not in {source_a, source_b}
                        or item["ingestion_run_id"] not in run_ids
                        or item["quote"] not in texts[0]
                        for item in evidence
                    ):
                        raise RuntimeError("Assertion evidence provenance is invalid")
                    attestations = [json.loads(item) for item in assertion["attestations"]]
                    if not attestations or any(
                        item["model"] != verifier.model or not item["supported"]
                        for item in attestations
                    ):
                        raise RuntimeError("Assertion verifier attestations did not persist")

                result = await session.run(
                    "MATCH ()-[r]->() WHERE r.id IN $ids "
                    "RETURN r.id AS id, startNode(r).id AS source_id, "
                    "endNode(r).id AS target_id, r.ingestion_run_ids AS runs, "
                    "r.evidence_references AS evidence, r.extraction_assertions AS assertions",
                    ids=list(relationship_ids),
                )
                persisted_edges = await result.data()
                if len(persisted_edges) != len(relationship_ids):
                    raise RuntimeError("Relationship identities are missing or duplicated")
                if any(
                    edge["source_id"] not in node_ids or edge["target_id"] not in node_ids
                    for edge in persisted_edges
                ):
                    raise RuntimeError("A relationship endpoint is outside the pilot batches")
                for edge in persisted_edges:
                    if not set(run_ids).issubset(set(edge["runs"])):
                        raise RuntimeError("Relationship ingestion runs did not accumulate")
                    edge_evidence = [json.loads(item) for item in edge["evidence"]]
                    if len(edge_evidence) != 2:
                        raise RuntimeError("Relationship replay duplicated or lost evidence")
                    for item in edge_evidence:
                        source_text = texts[0] if item["source_sha256"] == source_a else texts[1]
                        if item["quote"] not in source_text or item["ingestion_run_id"] not in run_ids:
                            raise RuntimeError("Relationship quote or run does not match source text")
                    edge_assertions = [json.loads(item) for item in edge["assertions"]]
                    relevant_edge_assertions = [
                        item for item in edge_assertions
                        if any(ref["source_sha256"] in {source_a, source_b}
                               for ref in item["evidence_references"])
                    ]
                    if len(relevant_edge_assertions) < 2 or any(
                        item["support_attestation"]["model"] != verifier.model
                        for item in relevant_edge_assertions
                    ):
                        raise RuntimeError("Relationship verifier metadata did not persist")

            return {
                "database": database,
                "extraction_model": extractor.model,
                "verifier_model": verifier.model,
                "source_documents": 2,
                "semantic_schema_constraints": semantic_schema["constraints_verified"],
                "preflight_semantic_nodes": preflight["semantic_nodes_to_label"],
                "admitted_chunks": len(first.screened.document.chunks) + len(second.screened.document.chunks),
                "nodes_per_document": [len(first.nodes), len(second.nodes)],
                "relationships_per_document": [len(first.relationships), len(second.relationships)],
                "first_write": first_write,
                "exact_replay": replay,
                "second_document_write": second_write,
                "shared_semantic_entities": len(shared_ids),
                "shared_semantic_labels": sorted({
                    node.node_type for node in first.nodes
                    if str(node.properties["id"]) in shared_ids
                }),
                "persisted_relationship_records": len(persisted_edges),
                "provenance_and_attestations_verified": True,
            }
    finally:
        try:
            fixture_node_ids = node_ids | assertion_ids
            if fixture_node_ids or relationship_ids:
                async with db.get_session() as session:
                    if fixture_node_ids:
                        await (await session.run(
                            "MATCH (a)-[r]->(b) WHERE a.id IN $ids OR b.id IN $ids DELETE r",
                            ids=list(fixture_node_ids),
                        )).consume()
                    if fixture_node_ids:
                        await (await session.run(
                            "MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
                            ids=list(fixture_node_ids),
                        )).consume()
                    result = await session.run(
                        "MATCH (n) WHERE n.id IN $ids RETURN count(n) AS count",
                        ids=list(fixture_node_ids),
                    )
                    if (await result.single())["count"]:
                        raise RuntimeError("Phase 5 pilot cleanup left graph nodes")
                    result = await session.run(
                        "MATCH (a)-[r]->(b) WHERE a.id IN $ids OR b.id IN $ids "
                        "RETURN count(r) AS count",
                        ids=list(fixture_node_ids),
                    )
                    if (await result.single())["count"]:
                        raise RuntimeError("Phase 5 pilot cleanup left graph relationships")
        finally:
            await extractor_client.client.close()
            await verifier.aclose()
            await db.close()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_pilot()), indent=2))
