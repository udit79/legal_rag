"""Typed state carried through the CPCR classification graph."""

from typing import TypedDict,NotRequired

from src.ingestion.continuity_cdm_contracts import CDMClassificationResult
from src.ingestion.continuity_contracts import ContinuityCandidate
from src.ingestion.continuity_gateway import GatewayResult


class CPCRState(TypedDict):
    candidate: ContinuityCandidate
    source_text: str
    target_text: str
    gateway_result: GatewayResult
    cdm_result: NotRequired[CDMClassificationResult]
    classification_rejection: NotRequired[str]
