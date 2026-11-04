import pytest
from pydantic import ValidationError

from src.ingestion.chunk_contracts import Chunk, ChunkSource, ChunkedDocument
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
    evaluate_candidate,
)

CHUNKING_VERSION = "blp-1.3.0"


def make_source(page: int, text: str, block_type: str) -> ChunkSource:
    return ChunkSource(
        page_number=page,
        block_id=f"page-{page}-block-0",
        block_reading_order=0,
        block_type=block_type,
        source_text_length=len(text),
        start_offset=0,
        end_offset=len(text),
    )


def make_pair(
    source_text: str,
    target_text: str,
    source_type: str = "TEXT",
    target_type: str = "TEXT",
    source_page: int = 1,
    target_page: int = 2,
) -> tuple[ContinuityCandidate, ChunkedDocument]:
    chunks = (
        Chunk(
            index=0,
            text=source_text,
            sources=(make_source(source_page, source_text, source_type),),
        ),
        Chunk(
            index=1,
            text=target_text,
            sources=(make_source(target_page, target_text, target_type),),
        ),
    )
    document = ChunkedDocument(
        document_id="gateway-doc",
        source_page_count=target_page,
        policy_version=CHUNKING_VERSION,
        chunks=chunks,
    )
    candidate = ContinuityCandidate(
        candidate_id=deterministic_candidate_id(
            document.document_id, CHUNKING_VERSION, 0, 1
        ),
        source=ChunkReference(
            document_id=document.document_id,
            chunk_index=0,
            page_number=source_page,
        ),
        target=ChunkReference(
            document_id=document.document_id,
            chunk_index=1,
            page_number=target_page,
        ),
        chunking_policy_version=CHUNKING_VERSION,
        candidate_policy_version="cpcr-candidate-1",
    )
    return candidate, document


def test_explicit_matching_table_continuation_is_accepted() -> None:
    candidate, document = make_pair(
        "Table 4 — Annual totals\n2025 | 10",
        "Table 4 (continued)\n2026 | 12",
        "TABLE",
        "TABLE",
    )

    result = evaluate_candidate(candidate, document)

    assert result.decision is GatewayDecision.ACCEPT
    assert result.relation_type is ContinuityRelation.TABLE_CONTINUATION
    assert result.evidence_codes == ("EXPLICIT_MATCHING_TABLE_CONTINUATION_CAPTION",)


def test_in_text_table_reference_does_not_establish_source_caption() -> None:
    candidate, document = make_pair(
        "The figures are explained in Table 4.",
        "Table 4 (continued)\n2026 | 12",
        "TABLE",
        "TABLE",
    )

    result = evaluate_candidate(candidate, document)

    assert result.decision is GatewayDecision.DEFER
    assert result.relation_type is None


@pytest.mark.parametrize(
    ("source_text", "target_text", "source_type", "target_type"),
    [
        ("Table 4 — Annual totals", "Table 5 (continued)\nrow", "TABLE", "TABLE"),
        ("Table 4 — Annual totals", "Table 4 (continued)\nrow", "TABLE", "TEXT"),
    ],
)
def test_weak_or_incompatible_table_evidence_defers(
    source_text: str,
    target_text: str,
    source_type: str,
    target_type: str,
) -> None:
    candidate, document = make_pair(
        source_text, target_text, source_type, target_type
    )

    result = evaluate_candidate(candidate, document)

    assert result.decision is GatewayDecision.DEFER
    assert result.relation_type is None


def test_explicit_soft_hyphen_word_continuation_is_accepted() -> None:
    candidate, document = make_pair("inter\u00ad", "national economic growth")

    result = evaluate_candidate(candidate, document)

    assert result.decision is GatewayDecision.ACCEPT
    assert result.relation_type is ContinuityRelation.SAME_PARAGRAPH
    assert result.evidence_codes == ("SOFT_HYPHEN_WORD_CONTINUATION",)


def test_text_like_other_block_can_use_soft_hyphen_evidence() -> None:
    candidate, document = make_pair(
        "inter\u00ad", "national economic growth", "OTHER", "OTHER"
    )

    result = evaluate_candidate(candidate, document)

    assert result.decision is GatewayDecision.ACCEPT
    assert result.relation_type is ContinuityRelation.SAME_PARAGRAPH


@pytest.mark.parametrize(
    ("source_text", "target_text"),
    [
        ("The provision remains incom-", "plete in the next paragraph."),
        ("The provision remains incom\u00ad", "Complete in the next paragraph."),
        ("The provision remains unfinished", "in the next paragraph."),
    ],
)
def test_ambiguous_textual_continuation_defers(
    source_text: str, target_text: str
) -> None:
    candidate, document = make_pair(source_text, target_text)

    result = evaluate_candidate(candidate, document)

    assert result.decision is GatewayDecision.DEFER
    assert result.relation_type is None


def test_completed_sentence_before_strong_section_marker_is_rejected() -> None:
    candidate, document = make_pair(
        "The preceding discussion ends here.", "Section 5 — New provision"
    )

    result = evaluate_candidate(candidate, document)

    assert result.decision is GatewayDecision.REJECT
    assert result.relation_type is ContinuityRelation.NO_CONTINUATION
    assert result.evidence_codes == ("COMPLETED_SENTENCE_BEFORE_STRONG_PROVISION",)


def test_matching_printed_page_number_is_not_a_provision_marker() -> None:
    candidate, document = make_pair(
        "The preceding discussion ends here.", "12", source_page=11, target_page=12
    )

    result = evaluate_candidate(candidate, document)

    assert result.decision is GatewayDecision.DEFER
    assert result.relation_type is None


def test_printed_page_number_is_ignored_before_real_provision_marker() -> None:
    candidate, document = make_pair(
        "The preceding discussion ends here.",
        "12\nSection 5 — New provision",
        source_page=11,
        target_page=12,
    )

    result = evaluate_candidate(candidate, document)

    assert result.decision is GatewayDecision.REJECT
    assert result.relation_type is ContinuityRelation.NO_CONTINUATION


@pytest.mark.parametrize(
    ("source_text", "target_text"),
    [
        ("The preceding discussion ends here.", "(a) A weak list label"),
        ("The preceding discussion continues", "Section 5 — A heading"),
    ],
)
def test_weak_or_incomplete_section_signals_defer(
    source_text: str, target_text: str
) -> None:
    candidate, document = make_pair(source_text, target_text)

    result = evaluate_candidate(candidate, document)

    assert result.decision is GatewayDecision.DEFER
    assert result.relation_type is None


def test_evaluation_is_deterministic_and_always_has_evidence() -> None:
    candidate, document = make_pair("Unfinished sentence", "continues here")

    first = evaluate_candidate(candidate, document)
    second = evaluate_candidate(candidate, document)

    assert first == second
    assert first.evidence_codes
    assert first.gateway_policy_version == CONTINUITY_GATEWAY_POLICY_VERSION


@pytest.mark.parametrize(
    ("decision", "relation"),
    [
        (GatewayDecision.ACCEPT, ContinuityRelation.NO_CONTINUATION),
        (GatewayDecision.REJECT, ContinuityRelation.SAME_PARAGRAPH),
        (GatewayDecision.DEFER, ContinuityRelation.UNKNOWN),
    ],
)
def test_gateway_result_rejects_invalid_decision_relation_pairs(
    decision: GatewayDecision, relation: ContinuityRelation
) -> None:
    with pytest.raises(ValidationError):
        GatewayResult(
            decision=decision,
            relation_type=relation,
            evidence_codes=("TEST",),
            gateway_policy_version=CONTINUITY_GATEWAY_POLICY_VERSION,
        )


def test_gateway_result_requires_evidence_code() -> None:
    with pytest.raises(ValidationError):
        GatewayResult(
            decision=GatewayDecision.DEFER,
            relation_type=None,
            evidence_codes=(),
            gateway_policy_version=CONTINUITY_GATEWAY_POLICY_VERSION,
        )
