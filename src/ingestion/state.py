from typing import Literal, NotRequired, TypedDict
from src.ingestion.contracts import DocumentFeatures, DocumentRef, RouteDecision

WorkflowStage = Literal[
    "PREFLIGHT",
    "ROUTING",
    "PROCESSING",
    "COMPLETE",
    "FAILED",
]
ErrorStage = Literal["PREFLIGHT", "ROUTING", "PROCESSING"]


class WorkflowError(TypedDict):
    code: str
    stage: ErrorStage
    message: NotRequired[str]


class IngestionState(TypedDict):
    document_ref: DocumentRef
    current_stage: WorkflowStage
    features: NotRequired[DocumentFeatures]
    route_decision: NotRequired[RouteDecision]
    parsed_artifact_ref: NotRequired[str]
    error: NotRequired[WorkflowError]
