"""Conservative deterministic routing for structural continuity candidates."""

import re
from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ingestion.chunk_contracts import Chunk, ChunkedDocument
from src.ingestion.chunking import STRONG_PROVISION_MARKER_PATTERNS
from src.ingestion.continuity_contracts import (
    ChunkReference,
    ContinuityCandidate,
    ContinuityRelation,
    validate_candidate_against_document,
)
from src.ingestion.contracts import NonEmptyStr

CONTINUITY_GATEWAY_POLICY_VERSION = "cpcr-gateway-1.0.0"


class GatewayDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    DEFER = "DEFER"


class GatewayResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: GatewayDecision
    relation_type: ContinuityRelation | None
    evidence_codes: tuple[NonEmptyStr, ...] = Field(min_length=1)
    gateway_policy_version: NonEmptyStr

    @model_validator(mode="after")
    def validate_decision_relation(self) -> Self:
        if self.decision == GatewayDecision.ACCEPT and self.relation_type not in {
            ContinuityRelation.SAME_PARAGRAPH,
            ContinuityRelation.TABLE_CONTINUATION,
        }:
            raise ValueError("ACCEPT requires a positive continuity relation")
        if (
            self.decision == GatewayDecision.REJECT
            and self.relation_type != ContinuityRelation.NO_CONTINUATION
        ):
            raise ValueError("REJECT must map to NO_CONTINUATION")
        if self.decision == GatewayDecision.DEFER and self.relation_type is not None:
            raise ValueError("DEFER must not have a final continuity relation")
        return self


def _chunk_for_reference(
    document: ChunkedDocument,
    reference: ChunkReference,
) -> Chunk:
    return next(chunk for chunk in document.chunks if chunk.index == reference.chunk_index)


def _only_block_type(chunk: Chunk, block_type: str) -> bool:
    return bool(chunk.sources) and all(
        source.block_type == block_type for source in chunk.sources
    )


def _text_block_group(chunk: Chunk) -> bool:
    return bool(chunk.sources) and all(
        source.block_type in {"TEXT", "OTHER"} for source in chunk.sources
    )


def _first_nonblank_line(text: str, page_number: int) -> str:
    lines = re.split(r"\r\n|\r|\n", text)
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Some parsers place a printed folio at the start of the next-page
        # chunk. Ignore it only when it exactly matches that physical page.
        if stripped.isascii() and stripped.isdecimal() and int(stripped) == page_number:
            continue
        return line
    return ""


def _starts_strong_provision(text: str, page_number: int) -> bool:
    first_line = _first_nonblank_line(text, page_number)
    return any(
        re.match(pattern, first_line) is not None
        for pattern in STRONG_PROVISION_MARKER_PATTERNS
    )


def _ends_at_sentence_boundary(text: str) -> bool:
    return re.search(r"[.!?।][\"'”’»)\]}]*\s*$", text) is not None


def _explicit_table_continuation(source_text: str, target_text: str) -> bool:
    source_ids = {
        match.group(1).casefold()
        for match in re.finditer(
            r"(?im)^\s*(?:table|schedule)\s+"
            r"([A-Z0-9][A-Z0-9.-]*)\b",
            source_text,
        )
    }
    target_continuations = {
        match.group(1).casefold()
        for match in re.finditer(
            r"(?im)^\s*(?:table|schedule)\s+([A-Z0-9][A-Z0-9.-]*)"
            r"\s*\(continued\)",
            target_text,
        )
    }
    return bool(source_ids & target_continuations)


def _result(
    decision: GatewayDecision,
    relation: ContinuityRelation | None,
    evidence_code: str,
) -> GatewayResult:
    return GatewayResult(
        decision=decision,
        relation_type=relation,
        evidence_codes=(evidence_code,),
        gateway_policy_version=CONTINUITY_GATEWAY_POLICY_VERSION,
    )


def evaluate_candidate(
    candidate: ContinuityCandidate,
    document: ChunkedDocument,
) -> GatewayResult:
    """Route a validated page-boundary candidate without semantic model calls."""
    validate_candidate_against_document(candidate, document)
    source_chunk = _chunk_for_reference(document, candidate.source)
    target_chunk = _chunk_for_reference(document, candidate.target)

    if _only_block_type(source_chunk, "TABLE") and _only_block_type(
        target_chunk, "TABLE"
    ):
        if _explicit_table_continuation(source_chunk.text, target_chunk.text):
            return _result(
                GatewayDecision.ACCEPT,
                ContinuityRelation.TABLE_CONTINUATION,
                "EXPLICIT_MATCHING_TABLE_CONTINUATION_CAPTION",
            )
        return _result(
            GatewayDecision.DEFER, None, "TABLE_CONTINUATION_NOT_EXPLICIT"
        )

    if _text_block_group(source_chunk) and _text_block_group(target_chunk):
        source_tail = source_chunk.text.rstrip()
        target_head = target_chunk.text.lstrip()
        if source_tail.endswith("\u00ad") and target_head[:1].islower():
            return _result(
                GatewayDecision.ACCEPT,
                ContinuityRelation.SAME_PARAGRAPH,
                "SOFT_HYPHEN_WORD_CONTINUATION",
            )
        if _ends_at_sentence_boundary(source_chunk.text) and _starts_strong_provision(
            target_chunk.text, candidate.target.page_number
        ):
            return _result(
                GatewayDecision.REJECT,
                ContinuityRelation.NO_CONTINUATION,
                "COMPLETED_SENTENCE_BEFORE_STRONG_PROVISION",
            )

    return _result(
        GatewayDecision.DEFER,
        None,
        "INSUFFICIENT_OR_AMBIGUOUS_CONTINUITY_EVIDENCE",
    )
