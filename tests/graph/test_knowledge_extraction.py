import asyncio
import json

import pytest

from src.domain.ontology import EpistemicMode
from src.graph.knowledge_extraction import (
    ExtractionProposal,
    ExtractionRejected,
    OpenAIStructuredExtractor,
    ProposedEntity,
    ProposedRelation,
    StructuredExtractionConfig,
    extract_validated_batch,
    proposal_to_batch,
)
from src.graph.graph_write_gate import EvidenceSpan
from src.graph.knowledge_writer import GraphIdentityConflict, persist_approved_batch
from tests.graph.test_graph_write_gate import _screened


def _proposal(text):
    span = EvidenceSpan(chunk_index=0, start_offset=0, end_offset=len(text), quote=text)
    return ExtractionProposal(
        entities=(
            ProposedEntity(ref="org", node_type="ORGANIZATION", properties={"name": "Court"}, epistemic_mode=EpistemicMode.VERBATIM_EXTRACTION, confidence=.95, evidence_spans=(span,)),
            ProposedEntity(ref="app", node_type="APPLICATION", properties={"name": "Filing"}, epistemic_mode=EpistemicMode.VERBATIM_EXTRACTION, confidence=.95, evidence_spans=(span,)),
        ),
        relationships=(ProposedRelation(source_ref="org", relation_type="OWNS", target_ref="app", epistemic_mode=EpistemicMode.VERBATIM_EXTRACTION, confidence=.95, evidence_spans=(span,)),),
    )


def test_proposal_maps_to_stable_ids_and_gate_approved_batch():
    text = "The Court owns the filing application."
    doc, screened = _screened(text)
    first = proposal_to_batch(_proposal(text), document=doc, screened=screened)
    second = proposal_to_batch(_proposal(text), document=doc, screened=screened)
    assert [n.properties["id"] for n in first.nodes] == [n.properties["id"] for n in second.nodes]
    assert first.relationships[0].id == second.relationships[0].id


def test_unsupported_quote_is_rejected_by_graph_write_gate():
    text = "A policy exists."
    doc, screened = _screened(text)
    proposal = ExtractionProposal(entities=(ProposedEntity(
        ref="org", node_type="ORGANIZATION", properties={"name": "Court"},
        epistemic_mode=EpistemicMode.VERBATIM_EXTRACTION, confidence=.95,
        evidence_spans=(EvidenceSpan(chunk_index=0, start_offset=0, end_offset=3, quote="Wrong"),),
    ),), relationships=())
    with pytest.raises(ValueError, match="does not match"):
        proposal_to_batch(proposal, document=doc, screened=screened)


def test_low_confidence_is_rejected_before_write():
    text = "The Court owns the filing application."
    doc, screened = _screened(text)
    proposal = _proposal(text)
    low = proposal.entities[0].model_copy(update={"confidence": .2})
    proposal = proposal.model_copy(update={"entities": (low, *proposal.entities[1:])})
    with pytest.raises(ExtractionRejected, match="confidence"):
        proposal_to_batch(proposal, document=doc, screened=screened)


def test_inferred_claim_is_never_persisted_as_a_graph_fact():
    text = "The Court owns the filing application."
    doc, screened = _screened(text)
    entity = _proposal(text).entities[0].model_copy(update={"epistemic_mode": EpistemicMode.LLM_INFERRED})
    with pytest.raises(ValueError, match="not admitted as graph facts"):
        proposal_to_batch(ExtractionProposal(entities=(entity,), relationships=()),
                          document=doc, screened=screened)


def test_model_cannot_invent_graph_identity_or_unknown_relationship_endpoint():
    entity = ProposedEntity(ref="x", node_type="ORGANIZATION",
                            properties={"id": "model-id", "name": "Court"},
                            epistemic_mode=EpistemicMode.VERBATIM_EXTRACTION,
                            confidence=.9, evidence_spans=(EvidenceSpan(
                                chunk_index=0, start_offset=0, end_offset=1, quote="x"),))
    doc, screened = _screened("x")
    with pytest.raises(ExtractionRejected, match="assign graph identities"):
        proposal_to_batch(ExtractionProposal(entities=(entity,), relationships=()),
                          document=doc, screened=screened)


def test_openai_adapter_uses_strict_schema_and_respects_token_ceiling():
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"choices": [type("Choice", (), {
                "message": type("Message", (), {"content": '{"entities":[],"relationships":[]}'}),
            })()]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    adapter = OpenAIStructuredExtractor(
        StructuredExtractionConfig(provider="openai", model="test-model", max_output_tokens=100),
        client=client,
    )
    output = asyncio.run(adapter.extract(payload={"chunks": []}, max_output_tokens=50))
    assert json.loads(output) == {"entities": [], "relationships": []}
    schema = captured["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    with pytest.raises(ValueError, match="exceed"):
        asyncio.run(adapter.extract(payload={}, max_output_tokens=101))


class FakeExtractor:
    async def extract(self, *, payload, max_output_tokens):
        assert max_output_tokens == 900
        text = payload["chunks"][0]["raw_text"]
        return json.dumps(_proposal(text).model_dump(mode="json"))


def test_extractor_output_is_parsed_and_validated():
    text = "The Court owns the filing application."
    doc, screened = _screened(text)
    batch = asyncio.run(extract_validated_batch(FakeExtractor(), document=doc, screened=screened, ingestion_run_id="run-1", max_output_tokens=900))
    assert len(batch.nodes) == 2 and len(batch.relationships) == 1


class FakeResult:
    def __init__(self, records): self.records = records
    def __aiter__(self): return self._iterate()
    async def _iterate(self):
        for row in self.records: yield row


class FakeTx:
    def __init__(self, *, conflict=False):
        self.conflict = conflict
    async def run(self, query, **params):
        row = params["rows"][0]
        if "MATCH (s" in query:
            return FakeResult([{"id": row["id"], "properties": row["properties"], "source_id": row["source_id"], "target_id": row["target_id"]}])
        props = dict(row["properties"])
        if self.conflict:
            props["name"] = "Conflicting name"
        return FakeResult([{"id": row["id"], "properties": props}])


class FakeSession:
    def __init__(self, *, conflict=False):
        self.conflict = conflict
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return None
    async def execute_write(self, fn, batch): return await fn(FakeTx(conflict=self.conflict), batch)


class FakeDb:
    def __init__(self, *, conflict=False):
        self.conflict = conflict
    def get_session(self): return FakeSession(conflict=self.conflict)


def test_writer_uses_one_transaction_after_gate_validation():
    text = "The Court owns the filing application."
    doc, screened = _screened(text)
    batch = proposal_to_batch(_proposal(text), document=doc, screened=screened).model_copy(update={"ingestion_run_id": "run-1"})
    assert asyncio.run(persist_approved_batch(FakeDb(), batch)) == {"nodes": 2, "relationships": 1}


def test_writer_refuses_unvalidated_object():
    with pytest.raises(TypeError):
        asyncio.run(persist_approved_batch(FakeDb(), object()))


def test_writer_rejects_existing_node_identity_conflict():
    text = "The Court owns the filing application."
    doc, screened = _screened(text)
    batch = proposal_to_batch(_proposal(text), document=doc, screened=screened).model_copy(update={"ingestion_run_id": "run-1"})
    with pytest.raises(GraphIdentityConflict, match="conflicting properties"):
        asyncio.run(persist_approved_batch(FakeDb(conflict=True), batch))
