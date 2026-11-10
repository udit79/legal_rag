from typing import Protocol
from langgraph.graph import END, START, StateGraph
from src.ingestion.chunk_contracts import ChunkedDocument
from src.ingestion.continuity_cdm_contracts import CDMClassificationRequest,CDMClassificationResult
from src.ingestion.continuity_gateway import GatewayDecision
from src.ingestion.continuity_contracts import AssessmentMethod,ContinuityAssessment,ContinuityRelation,validate_assessment_against_document
from src.ingestion.continuity_state import CPCRState

class ClassificationClient(Protocol):
    async def classify(
        self,
        request: CDMClassificationRequest,
    ) -> CDMClassificationResult: ...

async def scaledown_classification_node(
    state: CPCRState,
    client: ClassificationClient,
    document: ChunkedDocument,
) -> dict[str, CDMClassificationResult | str]:
    """Classify the supplied candidate using an injected ScaleDown client."""
    if state["gateway_result"].decision != GatewayDecision.DEFER:
        raise ValueError("ScaleDown classification is only allowed for DEFER candidates")
    request = CDMClassificationRequest(
        candidate_id=state["candidate"].candidate_id,
        source=state["candidate"].source,
        target=state["candidate"].target,
        source_text=state["source_text"],
        target_text=state["target_text"],
    )
    result = await client.classify(request)
    if result.candidate_id != state["candidate"].candidate_id:
        raise ValueError("ScaleDown result candidate ID does not match request")
    if result.relation_type == ContinuityRelation.TABLE_CONTINUATION:
        table_only = {"TABLE"}
        chunks_by_index = {
            chunk.index: chunk for chunk in document.chunks
        }
        source_chunk = chunks_by_index[state["candidate"].source.chunk_index]
        target_chunk = chunks_by_index[state["candidate"].target.chunk_index]
        if any(
            not chunk.sources
            or any(source.block_type not in table_only for source in chunk.sources)
            for chunk in (source_chunk, target_chunk)
        ):
            return {
                "cdm_result": result,
                "classification_rejection": "TABLE_CONTINUATION_REQUIRES_TABLE_CHUNKS",
            }
    assessment = ContinuityAssessment(
        candidate=state["candidate"],
        relation_type=result.relation_type,
        method=AssessmentMethod.CDM_CLASSIFY,
        evidence_codes=("SCALEDOWN_CLASSIFICATION",),
        assessment_policy_version="cpcr-cdm-1.0.0",
    )
    validate_assessment_against_document(assessment, document)
    return {"cdm_result": result}

def route_after_gateway(state: CPCRState) -> str:
    """Route only deferred gateway decisions to ScaleDown classification."""
    if state["gateway_result"].decision == GatewayDecision.DEFER:
        return "classify"
    return "finish"

def build_cpcr_graph(client: ClassificationClient, document: ChunkedDocument):
    """Compile the CPCR workflow with its ScaleDown client injected."""
    workflow = StateGraph(CPCRState)

    async def classify(
        state: CPCRState,
    ) -> dict[str, CDMClassificationResult | str]:
        return await scaledown_classification_node(state, client, document)

    workflow.add_node("classify", classify)
    workflow.add_conditional_edges(
        START,
        route_after_gateway,
        {"classify": "classify", "finish": END},
    )
    workflow.add_edge("classify", END)
    return workflow.compile()
