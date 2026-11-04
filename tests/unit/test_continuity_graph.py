import asyncio
import pytest

from src.ingestion.chunk_contracts import Chunk, ChunkSource, ChunkedDocument
from src.ingestion.continuity_cdm_contracts import (
    CDMClassificationRequest,
    CDMClassificationResult,
)
from src.ingestion.continuity_contracts import (
    ChunkReference,
    ContinuityCandidate,
    ContinuityRelation,
    deterministic_candidate_id,
)
from src.ingestion.continuity_gateway import (
    CONTINUITY_GATEWAY_POLICY_VERSION,
    GatewayDecision,
    GatewayResult,
)
from src.ingestion.continuity_graph import (
    build_cpcr_graph,
    route_after_gateway,
    scaledown_classification_node,
)
from src.ingestion.continuity_state import CPCRState


class StubScaleDownClient:
    def __init__(self) -> None:
        self.requests: list[CDMClassificationRequest] = []
        self.relation_type = ContinuityRelation.SAME_PARAGRAPH

    async def classify(
        self, request: CDMClassificationRequest
    ) -> CDMClassificationResult:
        self.requests.append(request)
        relation = self.relation_type
        probabilities = {item: 0.1 for item in ContinuityRelation}
        probabilities[relation] = 0.7
        return CDMClassificationResult(
            candidate_id=request.candidate_id,
            relation_type=relation,
            probabilities=probabilities,
            model_id="classify-1",
        )


def cpcr_state() -> CPCRState:
    candidate = ContinuityCandidate(
        candidate_id=deterministic_candidate_id("doc-1", "blp-1", 0, 1),
        source=ChunkReference(document_id="doc-1", chunk_index=0, page_number=4),
        target=ChunkReference(document_id="doc-1", chunk_index=1, page_number=5),
        chunking_policy_version="blp-1",
        candidate_policy_version="cpcr-candidate-1",
    )
    gateway_result = GatewayResult(
        decision=GatewayDecision.DEFER,
        relation_type=None,
        evidence_codes=("INSUFFICIENT_CONTEXT",),
        gateway_policy_version=CONTINUITY_GATEWAY_POLICY_VERSION,
    )
    return {
        "candidate": candidate,
        "source_text": "source exact text ",
        "target_text": " target exact text",
        "gateway_result": gateway_result,
        "cdm_result": None,
    }


def cpcr_document(
    source_type: str = "TEXT", target_type: str | None = None
) -> ChunkedDocument:
    source_text = "source exact text "
    target_text = " target exact text"

    def source(page: int, text: str, block_type: str) -> ChunkSource:
        return ChunkSource(
            page_number=page,
            block_id=f"page-{page}-block-0",
            block_reading_order=0,
            block_type=block_type,
            source_text_length=len(text),
            start_offset=0,
            end_offset=len(text),
        )

    return ChunkedDocument(
        document_id="doc-1",
        source_page_count=5,
        policy_version="blp-1",
        chunks=(
            Chunk(
                index=0,
                text=source_text,
                sources=(source(4, source_text, source_type),),
            ),
            Chunk(
                index=1,
                text=target_text,
                sources=(source(5, target_text, target_type or source_type),),
            ),
        ),
    )


def test_scaledown_node_calls_injected_client_and_returns_result_update() -> None:
    client = StubScaleDownClient()
    state = cpcr_state()
    document = cpcr_document()

    update = asyncio.run(scaledown_classification_node(state, client, document))

    assert len(client.requests) == 1
    sent = client.requests[0]
    assert sent.candidate_id == state["candidate"].candidate_id
    assert sent.source_text == "source exact text "
    assert sent.target_text == " target exact text"
    assert update["cdm_result"].relation_type is ContinuityRelation.SAME_PARAGRAPH


def test_scaledown_node_rejects_non_deferred_gateway_decision() -> None:
    client = StubScaleDownClient()

    with pytest.raises(
        ValueError,
        match="ScaleDown classification is only allowed for DEFER candidates",
    ):
        asyncio.run(
            scaledown_classification_node(
                state_with_gateway(GatewayDecision.ACCEPT), client, cpcr_document()
            )
        )

    assert client.requests == []


def state_with_gateway(
    decision: GatewayDecision,
) -> CPCRState:
    state = cpcr_state()
    relation = {
        GatewayDecision.ACCEPT: ContinuityRelation.SAME_PARAGRAPH,
        GatewayDecision.REJECT: ContinuityRelation.NO_CONTINUATION,
        GatewayDecision.DEFER: None,
    }[decision]
    state["gateway_result"] = GatewayResult(
        decision=decision,
        relation_type=relation,
        evidence_codes=("ROUTE_TEST",),
        gateway_policy_version=CONTINUITY_GATEWAY_POLICY_VERSION,
    )
    return state


def test_route_after_gateway_classifies_only_defer() -> None:
    assert route_after_gateway(state_with_gateway(GatewayDecision.DEFER)) == "classify"
    assert route_after_gateway(state_with_gateway(GatewayDecision.ACCEPT)) == "finish"
    assert route_after_gateway(state_with_gateway(GatewayDecision.REJECT)) == "finish"


def test_graph_sends_deferred_candidate_to_client_exactly_once() -> None:
    client = StubScaleDownClient()
    graph = build_cpcr_graph(client, cpcr_document())

    final_state = asyncio.run(graph.ainvoke(state_with_gateway(GatewayDecision.DEFER)))

    assert len(client.requests) == 1
    assert final_state["cdm_result"] is not None
    assert final_state["cdm_result"].candidate_id == client.requests[0].candidate_id


def test_graph_finishes_accept_and_reject_without_client_calls() -> None:
    client = StubScaleDownClient()
    graph = build_cpcr_graph(client, cpcr_document())

    accepted_state = asyncio.run(
        graph.ainvoke(state_with_gateway(GatewayDecision.ACCEPT))
    )
    rejected_state = asyncio.run(
        graph.ainvoke(state_with_gateway(GatewayDecision.REJECT))
    )

    assert client.requests == []
    assert accepted_state["cdm_result"] is None
    assert rejected_state["cdm_result"] is None


def test_graph_rejects_table_relation_for_text_sources() -> None:
    client = StubScaleDownClient()
    client.relation_type = ContinuityRelation.TABLE_CONTINUATION
    graph = build_cpcr_graph(client, cpcr_document())

    with pytest.raises(ValueError, match="TABLE-only"):
        asyncio.run(graph.ainvoke(state_with_gateway(GatewayDecision.DEFER)))


def test_graph_rejects_table_relation_for_footnote_to_table_candidate() -> None:
    client = StubScaleDownClient()
    client.relation_type = ContinuityRelation.TABLE_CONTINUATION
    graph = build_cpcr_graph(client, cpcr_document("OTHER", "TABLE"))

    with pytest.raises(ValueError, match="TABLE-only"):
        asyncio.run(graph.ainvoke(state_with_gateway(GatewayDecision.DEFER)))

