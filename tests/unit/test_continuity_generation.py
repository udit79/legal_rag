import pytest
from pydantic import ValidationError

from src.ingestion.chunk_contracts import Chunk, ChunkSource, ChunkedDocument
from src.ingestion.continuity import generate_continuity_candidates
from src.ingestion.continuity_contracts import (
    deterministic_candidate_id,
    validate_candidate_against_document,
)


POLICY_VERSION = "blp-1.2.0"


def make_source(page_number: int) -> ChunkSource:
    return ChunkSource(
        page_number=page_number,
        block_id=f"page-{page_number}-block-0",
        block_reading_order=0,
        block_type="TEXT",
        source_text_length=4,
        start_offset=0,
        end_offset=4,
    )


def make_document(
    page_count: int, pages: tuple[int, ...]
) -> ChunkedDocument:
    return ChunkedDocument(
        document_id="doc-1",
        source_page_count=page_count,
        policy_version=POLICY_VERSION,
        chunks=tuple(
            Chunk(index=index, text="text", sources=(make_source(page),))
            for index, page in enumerate(pages)
        ),
    )


def test_generates_final_to_first_candidate_for_adjacent_pages() -> None:
    document = make_document(2, (1, 1, 2, 2))

    candidates = generate_continuity_candidates(document, "cpcr-candidate-1")

    assert len(candidates) == 1
    assert candidates[0].source.chunk_index == 1
    assert candidates[0].target.chunk_index == 2
    assert candidates[0].chunking_policy_version == POLICY_VERSION
    assert candidates[0].candidate_policy_version == "cpcr-candidate-1"
    assert candidates[0].candidate_id == deterministic_candidate_id(
        document.document_id, POLICY_VERSION, 1, 2
    )
    validate_candidate_against_document(candidates[0], document)


@pytest.mark.parametrize(
    ("page_count", "pages"),
    [
        (1, ()),
        (1, (1,)),
        (3, (1, 3)),
        (4, (1, 4)),
    ],
)
def test_missing_intermediate_pages_are_not_bridged(
    page_count: int, pages: tuple[int, ...]
) -> None:
    assert generate_continuity_candidates(
        make_document(page_count, pages), "cpcr-candidate-1"
    ) == ()


def test_repeated_generation_is_deterministic() -> None:
    document = make_document(3, (1, 2, 2, 3))

    first = generate_continuity_candidates(document, "cpcr-candidate-1")
    second = generate_continuity_candidates(document, "cpcr-candidate-1")

    assert first == second
    assert [candidate.candidate_id for candidate in first] == [
        candidate.candidate_id for candidate in second
    ]
    assert [(item.source.page_number, item.target.page_number) for item in first] == [
        (1, 2),
        (2, 3),
    ]


def test_invalid_candidate_policy_version_is_rejected_even_without_candidates() -> None:
    with pytest.raises(ValueError, match="candidate_policy_version"):
        generate_continuity_candidates(make_document(1, ()), " ")


def test_candidate_policy_version_rejects_surrounding_whitespace() -> None:
    with pytest.raises(ValueError, match="surrounding whitespace"):
        generate_continuity_candidates(make_document(2, (1, 2)), " cpcr-1 ")


def test_chunked_document_contract_rejects_duplicate_chunk_indexes() -> None:
    document = make_document(2, (1, 2))
    with pytest.raises(ValidationError, match="contiguous"):
        ChunkedDocument(
            document_id=document.document_id,
            source_page_count=document.source_page_count,
            policy_version=document.policy_version,
            chunks=(
                document.chunks[0],
                document.chunks[1].model_copy(update={"index": 0}),
            ),
        )


def test_candidate_generation_rejects_duplicate_indexes_in_malformed_document() -> None:
    document = make_document(2, (1, 2))
    malformed_document = document.model_copy(
        update={
            "chunks": (
                document.chunks[0],
                document.chunks[1].model_copy(update={"index": 0}),
            )
        }
    )

    with pytest.raises(ValueError, match="Duplicate chunk index detected: 0"):
        generate_continuity_candidates(malformed_document, "cpcr-candidate-1")


def test_empty_source_provenance_is_rejected() -> None:
    document = make_document(2, (1, 2))
    invalid_chunk = Chunk.model_construct(index=2, text="unattributed", sources=())
    malformed_document = document.model_copy(
        update={"chunks": (*document.chunks, invalid_chunk)}
    )

    with pytest.raises(ValueError, match="Chunk 2 has no source provenance"):
        generate_continuity_candidates(malformed_document, "cpcr-candidate-1")


def test_cross_page_provenance_is_rejected() -> None:
    document = make_document(2, (1, 2))
    cross_page_chunk = Chunk(
        index=2,
        text="cross-page",
        sources=(make_source(1), make_source(2)),
    )
    malformed_document = document.model_copy(
        update={"chunks": (*document.chunks, cross_page_chunk)}
    )

    with pytest.raises(ValueError, match="Chunk 2 has cross-page provenance"):
        generate_continuity_candidates(malformed_document, "cpcr-candidate-1")


def test_invalid_zero_page_provenance_is_rejected() -> None:
    document = make_document(2, (1, 2))
    invalid_source = ChunkSource.model_construct(
        page_number=0,
        block_id="page-0-block-0",
        block_reading_order=0,
        block_type="TEXT",
        source_text_length=4,
        start_offset=0,
        end_offset=4,
    )
    invalid_chunk = Chunk.model_construct(
        index=2, text="invalid page", sources=(invalid_source,)
    )
    malformed_document = document.model_copy(
        update={"chunks": (*document.chunks, invalid_chunk)}
    )

    with pytest.raises(ValueError, match="Chunk 2 references an invalid page number"):
        generate_continuity_candidates(malformed_document, "cpcr-candidate-1")
