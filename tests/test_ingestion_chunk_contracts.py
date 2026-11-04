import pytest
from pydantic import ValidationError
from src.ingestion.chunk_contracts import Chunk, ChunkSource, ChunkedDocument
from src.ingestion.parser_contracts import (
    CanonicalBlock,
    SourceCoordinates,
    canonical_block_id,
)


def source_data(
    *,
    page_number: int = 1,
    block_reading_order: int = 0,
    block_type: str = "TEXT",
    source_text_length: int = 20,
    start_offset: int = 0,
    end_offset: int = 10,
) -> dict[str, object]:
    return {
        "page_number": page_number,
        "block_id": f"page-{page_number}-block-{block_reading_order}",
        "block_reading_order": block_reading_order,
        "block_type": block_type,
        "source_text_length": source_text_length,
        "start_offset": start_offset,
        "end_offset": end_offset,
    }


def chunk_data(
    index: int = 0, sources: list[dict[str, object]] | None = None
) -> dict[str, object]:
    return {
        "index": index,
        "text": "Evidence text",
        "sources": sources if sources is not None else [source_data()],
    }


def chunked_document_data(
    chunks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "document_id": "doc-123",
        "source_page_count": 2,
        "policy_version": "chunk-v1",
        "chunks": chunks if chunks is not None else [chunk_data()],
    }


def test_chunked_document_accepts_ordered_multi_block_and_split_block_sources():
    chunks = [
        chunk_data(
            0,
            [
                source_data(block_reading_order=0, start_offset=0, end_offset=10),
                source_data(
                    block_reading_order=1,
                    source_text_length=12,
                    start_offset=0,
                    end_offset=5,
                ),
            ],
        ),
        chunk_data(
            1,
            [
                source_data(
                    block_reading_order=1,
                    source_text_length=12,
                    start_offset=5,
                    end_offset=12,
                )
            ],
        ),
    ]

    document = ChunkedDocument.model_validate(chunked_document_data(chunks))

    assert tuple(chunk.index for chunk in document.chunks) == (0, 1)
    assert len(document.chunks[0].sources) == 2
    assert document.chunks[0].sources[0].block_type == "TEXT"
    assert document.chunks[0].sources[0].block_id == "page-1-block-0"
    assert document.chunks[0].sources[1].block_reading_order == 1
    assert document.chunks[1].sources[0].start_offset == 5
    assert document.policy_version == "chunk-v1"


@pytest.mark.parametrize(
    ("start_offset", "end_offset", "source_text_length"),
    [
        (-1, 5, 10),
        (5, 5, 10),
        (6, 5, 10),
        (0, 11, 10),
    ],
)
def test_chunk_source_rejects_invalid_offsets(
    start_offset: int,
    end_offset: int,
    source_text_length: int,
):
    with pytest.raises(ValidationError):
        ChunkSource.model_validate(
            source_data(
                start_offset=start_offset,
                end_offset=end_offset,
                source_text_length=source_text_length,
            )
        )


def test_chunk_source_requires_offset_fields_as_a_group():
    data = source_data()
    data.pop("end_offset")

    with pytest.raises(ValidationError, match="provided together"):
        ChunkSource.model_validate(data)


@pytest.mark.parametrize("text", ["compliance", "é📜"])
def test_text_offsets_count_unicode_characters_with_half_open_bounds(text: str):
    source = ChunkSource.model_validate(
        source_data(
            source_text_length=len(text),
            start_offset=0,
            end_offset=len(text),
        )
    )

    assert source.start_offset is not None
    assert source.end_offset is not None
    assert source.source_text_length is not None
    assert text[source.start_offset : source.end_offset] == text
    assert source.source_text_length == len(text)
    if text == "compliance":
        assert (source.start_offset, source.end_offset) == (0, 10)
    else:
        assert len(text.encode("utf-8")) > source.source_text_length


def test_chunk_source_accepts_full_and_partial_block_ranges():
    full_source = ChunkSource.model_validate(
        source_data(source_text_length=10, start_offset=0, end_offset=10)
    )
    partial_source = ChunkSource.model_validate(
        source_data(source_text_length=10, start_offset=3, end_offset=7)
    )

    assert (full_source.start_offset, full_source.end_offset) == (0, 10)
    assert (partial_source.start_offset, partial_source.end_offset) == (3, 7)


def test_chunk_rejects_duplicate_or_overlapping_ranges_for_one_block():
    duplicate_ranges = [
        source_data(start_offset=0, end_offset=10),
        source_data(start_offset=0, end_offset=10),
    ]
    overlapping_ranges = [
        source_data(source_text_length=30, start_offset=0, end_offset=20),
        source_data(source_text_length=30, start_offset=10, end_offset=30),
    ]

    for ranges in (duplicate_ranges, overlapping_ranges):
        with pytest.raises(ValidationError, match="cannot overlap"):
            Chunk.model_validate(chunk_data(sources=ranges))


def test_chunk_accepts_adjacent_ranges_for_one_block():
    chunk = Chunk.model_validate(
        chunk_data(
            sources=[
                source_data(start_offset=0, end_offset=10),
                source_data(start_offset=10, end_offset=20),
            ]
        )
    )

    assert tuple(
        (source.start_offset, source.end_offset) for source in chunk.sources
    ) == ((0, 10), (10, 20))


def test_chunk_rejects_duplicate_image_source_without_offsets():
    image_source = {
        "page_number": 1,
        "block_id": "page-1-block-2",
        "block_reading_order": 2,
        "block_type": "IMAGE",
    }

    with pytest.raises(ValidationError, match="duplicate source reference"):
        Chunk.model_validate(chunk_data(sources=[image_source, image_source.copy()]))


@pytest.mark.parametrize(
    "second_source",
    [
        source_data(source_text_length=21, start_offset=10, end_offset=20),
        source_data(
            block_type="TABLE",
            source_text_length=20,
            start_offset=10,
            end_offset=20,
        ),
    ],
)
def test_chunk_rejects_repeated_block_id_with_inconsistent_metadata(
    second_source: dict[str, object],
):
    with pytest.raises(ValidationError, match="consistent block type"):
        Chunk.model_validate(
            chunk_data(
                sources=[
                    source_data(start_offset=0, end_offset=10),
                    second_source,
                ]
            )
        )


def test_same_block_can_contribute_non_overlapping_ranges_to_different_chunks():
    document = ChunkedDocument.model_validate(
        chunked_document_data(
            [
                chunk_data(0, [source_data(start_offset=0, end_offset=10)]),
                chunk_data(1, [source_data(start_offset=10, end_offset=20)]),
            ]
        )
    )

    assert (
        document.chunks[0].sources[0].block_id == document.chunks[1].sources[0].block_id
    )


def test_non_text_source_can_be_identified_without_text_offsets():
    source = ChunkSource.model_validate(
        {
            "page_number": 1,
            "block_id": "page-1-block-2",
            "block_reading_order": 2,
            "block_type": "IMAGE",
        }
    )

    assert source.block_type == "IMAGE"
    assert source.start_offset is None


@pytest.mark.parametrize("block_type", ["TEXT", "TABLE", "OTHER"])
def test_each_text_bearing_block_type_requires_source_offsets(block_type: str):
    data = source_data(block_type=block_type)
    data["source_text_length"] = None
    data["start_offset"] = None
    data["end_offset"] = None

    with pytest.raises(ValidationError, match="require a source text range"):
        ChunkSource.model_validate(data)


def test_vector_block_rejects_text_offsets():
    with pytest.raises(ValidationError, match="cannot have text offsets"):
        ChunkSource.model_validate(source_data(block_type="VECTOR"))


def test_image_block_may_carry_text_offsets_when_parser_provides_text():
    source = ChunkSource.model_validate(
        source_data(
            block_type="IMAGE",
            source_text_length=8,
            start_offset=0,
            end_offset=8,
        )
    )

    assert source.block_type == "IMAGE"
    assert (source.start_offset, source.end_offset) == (0, 8)


def test_chunk_rejects_empty_text():
    data = chunk_data()
    data["text"] = ""

    with pytest.raises(ValidationError, match="cannot be empty"):
        Chunk.model_validate(data)


def test_chunk_accepts_nonempty_whitespace_text():
    data = chunk_data()
    data["text"] = " \t\n"

    assert Chunk.model_validate(data).text == " \t\n"


def test_chunk_requires_source_provenance():
    data = chunk_data()
    data["sources"] = []

    with pytest.raises(ValidationError, match="at least one source"):
        Chunk.model_validate(data)


@pytest.mark.parametrize("index", [-1, True])
def test_chunk_rejects_invalid_index(index: int | bool):
    with pytest.raises(ValidationError):
        Chunk.model_validate(chunk_data(index=index))


def test_chunked_document_rejects_noncontiguous_indices():
    with pytest.raises(ValidationError, match="zero-based and contiguous"):
        ChunkedDocument.model_validate(
            chunked_document_data([chunk_data(0), chunk_data(2)])
        )


@pytest.mark.parametrize(
    "chunks",
    [
        [chunk_data(0), chunk_data(0)],
        [chunk_data(1)],
        [chunk_data(1), chunk_data(0)],
    ],
)
def test_chunked_document_rejects_duplicate_missing_or_out_of_order_indices(
    chunks: list[dict[str, object]],
):
    with pytest.raises(ValidationError, match="zero-based and contiguous"):
        ChunkedDocument.model_validate(chunked_document_data(chunks))


def test_chunk_rejects_sources_out_of_document_order():
    data = chunk_data(
        sources=[
            source_data(block_reading_order=1),
            source_data(block_reading_order=0),
        ]
    )

    with pytest.raises(ValidationError, match="document reading order"):
        Chunk.model_validate(data)


def test_chunked_document_rejects_provenance_outside_source_pages():
    data = chunked_document_data([chunk_data(sources=[source_data(page_number=3)])])

    with pytest.raises(ValidationError, match="outside the source"):
        ChunkedDocument.model_validate(data)


def test_chunk_source_requires_block_id_and_chunked_document_requires_document_id():
    source = source_data()
    source.pop("block_id")
    with pytest.raises(ValidationError):
        ChunkSource.model_validate(source)

    document = chunked_document_data()
    document.pop("document_id")
    with pytest.raises(ValidationError):
        ChunkedDocument.model_validate(document)


def test_chunk_source_allows_stable_identity_across_reading_order_repair():
    data = source_data(block_reading_order=2)
    data["block_id"] = "page-1-block-0"
    assert ChunkSource.model_validate(data).block_id == "page-1-block-0"

    data["block_id"] = "page-2-block-0"
    with pytest.raises(ValidationError, match="page-scoped"):
        ChunkSource.model_validate(data)


def test_canonical_block_id_is_stable_and_page_scoped():
    first = canonical_block_id(page_number=2, reading_order=3)
    repeated = canonical_block_id(page_number=2, reading_order=3)
    different_page = canonical_block_id(page_number=3, reading_order=3)

    assert first == repeated == "page-2-block-3"
    assert different_page != first


@pytest.mark.parametrize("policy_version", ["", "  "])
def test_chunked_document_rejects_missing_or_blank_policy_version(
    policy_version: str,
):
    data = chunked_document_data()
    data["policy_version"] = policy_version
    with pytest.raises(ValidationError):
        ChunkedDocument.model_validate(data)


def test_chunked_document_distinguishes_blocks_with_identical_text():
    first = CanonicalBlock(
        page_number=1,
        text="Same evidence",
        block_type="TEXT",
        reading_order=0,
        source_coordinates=SourceCoordinates(
            x0=0,
            y0=0,
            x1=10,
            y1=10,
            unit="PDF_POINT",
            origin="TOP_LEFT",
        ),
    )
    second = first.model_copy(
        update={"reading_order": 1, "block_id": "page-1-block-1"}
    )
    assert first.text == second.text
    assert first.block_id != second.block_id
    assert first.model_dump()["block_id"] == first.block_id
    assert second.model_dump()["block_id"] == second.block_id

    document = ChunkedDocument.model_validate(
        chunked_document_data(
            [
                chunk_data(
                    0,
                    [
                        source_data(block_reading_order=0),
                        source_data(block_reading_order=1),
                    ],
                )
            ]
        )
    )

    assert tuple(source.block_id for source in document.chunks[0].sources) == (
        first.block_id,
        second.block_id,
    )


def test_chunked_document_is_deterministic_for_identical_input():
    data = chunked_document_data(
        [
            chunk_data(
                0,
                [
                    source_data(block_type="TABLE", source_text_length=20),
                ],
            )
        ]
    )

    first = ChunkedDocument.model_validate(data)
    second = ChunkedDocument.model_validate(data)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
