import pytest
from pydantic import ValidationError

from src.ingestion.chunk_contracts import Chunk, ChunkSource, ChunkedDocument
from src.ingestion.continuity_contracts import (
    AssessmentMethod,
    ChunkReference,
    ContinuityAssessment,
    ContinuityCandidate,
    ContinuityRelation,
    deterministic_candidate_id,
    validate_assessment_against_document,
    validate_candidate_against_document,
)

CHUNKING_VERSION = "blp-1.2.0"


def reference(index: int, page: int, document_id: str = "doc-1") -> ChunkReference:
    return ChunkReference(
        document_id=document_id, chunk_index=index, page_number=page
    )


def candidate(
    source: ChunkReference | None = None,
    target: ChunkReference | None = None,
    candidate_id: str | None = None,
) -> ContinuityCandidate:
    source = source or reference(0, 4)
    target = target or reference(1, 5)
    return ContinuityCandidate(
        candidate_id=candidate_id
        or deterministic_candidate_id(
            source.document_id,
            CHUNKING_VERSION,
            source.chunk_index,
            target.chunk_index,
        ),
        source=source,
        target=target,
        chunking_policy_version=CHUNKING_VERSION,
        candidate_policy_version="cpcr-candidate-1",
    )


def source(page: int, block_type: str = "TEXT") -> ChunkSource:
    return ChunkSource(
        page_number=page,
        block_id=f"page-{page}-block-0",
        block_reading_order=0,
        block_type=block_type,
        source_text_length=4,
        start_offset=0,
        end_offset=4,
    )


def document(
    first_type: str = "TEXT",
    second_type: str = "TEXT",
    first_page: int = 4,
    second_page: int = 5,
    mixed_first_page: bool = False,
) -> ChunkedDocument:
    first_sources = (source(first_page, first_type),)
    if mixed_first_page:
        first_sources += (source(first_page + 1, first_type),)
    return ChunkedDocument(
        document_id="doc-1",
        source_page_count=max(first_page, second_page),
        policy_version=CHUNKING_VERSION,
        chunks=(
            Chunk(index=0, text="text", sources=first_sources),
            Chunk(index=1, text="text", sources=(source(second_page, second_type),)),
        ),
    )


def multi_chunk_page_document() -> ChunkedDocument:
    page_chunks = ((4, 0), (4, 1), (5, 2), (5, 3))
    return ChunkedDocument(
        document_id="doc-1",
        source_page_count=5,
        policy_version=CHUNKING_VERSION,
        chunks=tuple(
            Chunk(index=index, text="text", sources=(source(page),))
            for page, index in page_chunks
        ),
    )


def assessment(relation: ContinuityRelation) -> ContinuityAssessment:
    return ContinuityAssessment(
        candidate=candidate(),
        relation_type=relation,
        method=AssessmentMethod.DETERMINISTIC,
        evidence_codes=("TEST_EVIDENCE",),
        assessment_policy_version="cpcr-assessment-1",
    )


def test_valid_adjacent_page_candidate_and_repeat_id() -> None:
    first = candidate()
    second = candidate()

    assert first == second
    assert first.candidate_id == deterministic_candidate_id(
        "doc-1", CHUNKING_VERSION, 0, 1
    )
    validate_candidate_against_document(first, document())


def test_candidate_may_represent_nonterminal_chunks_in_distinct_content_flows() -> None:
    doc = multi_chunk_page_document()
    validate_candidate_against_document(
        candidate(reference(0, 4), reference(3, 5)), doc
    )


def test_page_collection_rejects_any_chunk_without_source_provenance() -> None:
    doc = document()
    empty_chunk = Chunk.model_construct(index=2, text="unused", sources=())
    doc_with_empty_chunk = doc.model_copy(
        update={"chunks": (*doc.chunks, empty_chunk)}
    )

    with pytest.raises(ValueError, match="Chunk 2 has no source provenance"):
        validate_candidate_against_document(candidate(), doc_with_empty_chunk)


def test_page_collection_rejects_cross_page_provenance_anywhere_in_document() -> None:
    doc = document()
    cross_page_chunk = Chunk(
        index=2,
        text="cross page",
        sources=(source(8), source(9)),
    )
    doc_with_cross_page_chunk = doc.model_copy(
        update={"chunks": (*doc.chunks, cross_page_chunk)}
    )

    with pytest.raises(ValueError, match="Chunk 2 has cross-page provenance"):
        validate_candidate_against_document(candidate(), doc_with_cross_page_chunk)


@pytest.mark.parametrize(
    ("source_ref", "target_ref"),
    [
        (reference(0, 4, "doc-a"), reference(1, 5, "doc-b")),
        (reference(0, 4), reference(1, 4)),
        (reference(0, 4), reference(1, 6)),
        (reference(1, 5), reference(0, 4)),
        (reference(0, 4), reference(0, 5)),
    ],
)
def test_invalid_candidate_transitions_are_rejected(
    source_ref: ChunkReference, target_ref: ChunkReference
) -> None:
    with pytest.raises(ValidationError):
        candidate(source_ref, target_ref)


def test_incorrect_candidate_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Candidate ID"):
        candidate(candidate_id="wrong")


def test_same_chunk_cannot_be_referenced_as_both_candidate_ends() -> None:
    same = reference(0, 4)
    with pytest.raises(ValidationError):
        candidate(same, same)


def test_candidate_ids_distinguish_ambiguous_concatenations() -> None:
    assert deterministic_candidate_id("doc", CHUNKING_VERSION, 12, 3) != (
        deterministic_candidate_id("doc1", CHUNKING_VERSION, 2, 3)
    )


@pytest.mark.parametrize(
    ("source_ref", "target_ref", "doc"),
    [
        (reference(2, 4), reference(3, 5), document()),
        (reference(0, 5), reference(1, 6), document()),
        (reference(0, 4), reference(2, 5), document()),
        (reference(0, 4, "other"), reference(1, 5, "other"), document()),
    ],
)
def test_document_reference_validation_rejects_invalid_references(
    source_ref: ChunkReference, target_ref: ChunkReference, doc: ChunkedDocument
) -> None:
    # Build a valid adjacent-page identity, then exercise document lookup and metadata.
    cand = candidate(source_ref, target_ref)
    with pytest.raises(ValueError):
        validate_candidate_against_document(cand, doc)


def test_candidate_validation_rejects_chunk_with_multiple_source_pages() -> None:
    with pytest.raises(ValueError, match="exactly one page"):
        validate_candidate_against_document(
            candidate(), document(mixed_first_page=True)
        )


def test_candidate_chunking_version_must_match_document() -> None:
    doc = document().model_copy(update={"policy_version": "blp-1.1.0"})
    with pytest.raises(ValueError, match="different chunking version"):
        validate_candidate_against_document(candidate(), doc)


def test_candidate_reference_page_cannot_exceed_document_page_count() -> None:
    cand = candidate(reference(0, 5), reference(1, 6))
    with pytest.raises(ValueError, match="exceeds document page count"):
        validate_candidate_against_document(
            cand, document(first_page=5, second_page=5)
        )


def test_chunked_document_rejects_non_contiguous_tuple_indexes() -> None:
    doc = document()
    with pytest.raises(ValidationError, match="contiguous"):
        ChunkedDocument(
            document_id=doc.document_id,
            source_page_count=doc.source_page_count,
            policy_version=doc.policy_version,
            chunks=(doc.chunks[0], doc.chunks[1].model_copy(update={"index": 2})),
        )


def test_table_continuation_requires_two_table_only_chunks() -> None:
    validate_assessment_against_document(
        assessment(ContinuityRelation.TABLE_CONTINUATION),
        document("TABLE", "TABLE"),
    )
    with pytest.raises(ValueError, match="TABLE-only"):
        validate_assessment_against_document(
            assessment(ContinuityRelation.TABLE_CONTINUATION),
            document("TABLE", "TEXT"),
        )


def test_table_continuation_rejects_footnote_to_table_sources() -> None:
    with pytest.raises(ValueError, match="TABLE-only"):
        validate_assessment_against_document(
            assessment(ContinuityRelation.TABLE_CONTINUATION),
            document("OTHER", "TABLE"),
        )


def test_same_paragraph_rejects_table_only_chunks() -> None:
    with pytest.raises(ValueError, match="TEXT or OTHER"):
        validate_assessment_against_document(
            assessment(ContinuityRelation.SAME_PARAGRAPH),
            document("TABLE", "TABLE"),
        )


def test_same_paragraph_rejects_image_sources_for_this_milestone() -> None:
    with pytest.raises(ValueError, match="TEXT or OTHER"):
        validate_assessment_against_document(
            assessment(ContinuityRelation.SAME_PARAGRAPH),
            document("IMAGE", "IMAGE"),
        )


def test_same_paragraph_with_text_sources_is_valid() -> None:
    validate_assessment_against_document(
        assessment(ContinuityRelation.SAME_PARAGRAPH), document("TEXT", "TEXT")
    )


@pytest.mark.parametrize(
    "relation",
    [ContinuityRelation.NO_CONTINUATION, ContinuityRelation.UNKNOWN],
)
def test_negative_and_unknown_assessments_are_not_positive_edges(
    relation: ContinuityRelation,
) -> None:
    result = assessment(relation)

    assert result.is_positive_outcome is False
    validate_assessment_against_document(result, document())


def test_unknown_requires_nonempty_evidence_code() -> None:
    with pytest.raises(ValidationError):
        ContinuityAssessment(
            candidate=candidate(),
            relation_type=ContinuityRelation.UNKNOWN,
            method=AssessmentMethod.DETERMINISTIC,
            evidence_codes=(),
            assessment_policy_version="cpcr-assessment-1",
        )


def test_unknown_with_insufficient_context_evidence_is_valid() -> None:
    result = assessment(ContinuityRelation.UNKNOWN).model_copy(
        update={"evidence_codes": ("INSUFFICIENT_CONTEXT",)}
    )
    assert result.relation_type is ContinuityRelation.UNKNOWN
    assert result.is_positive_outcome is False


def test_candidate_and_assessment_policies_are_independently_recorded() -> None:
    result = assessment(ContinuityRelation.UNKNOWN)
    assert result.candidate.candidate_policy_version == "cpcr-candidate-1"
    assert result.assessment_policy_version == "cpcr-assessment-1"


def test_chunk_reference_uses_strict_integer_fields() -> None:
    with pytest.raises(ValidationError):
        ChunkReference(document_id="doc-1", chunk_index=0.0, page_number=4)
