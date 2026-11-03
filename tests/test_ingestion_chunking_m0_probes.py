import pytest
from pydantic import ValidationError

from src.ingestion.chunk_contracts import Chunk, ChunkedDocument


def source(page=1, order=0, kind="TEXT", length=20, start=0, end=10):
    return {
        "page_number": page,
        "block_id": f"page-{page}-block-{order}",
        "block_reading_order": order,
        "block_type": kind,
        "source_text_length": length,
        "start_offset": start,
        "end_offset": end,
    }


def chunk(text="Evidence", index=0, sources=None):
    return {
        "index": index,
        "text": text,
        "sources": sources if sources is not None else [source()],
    }


def document(chunks=None, version="chunk-v1"):
    return {
        "document_id": "doc-1",
        "source_page_count": 2,
        "policy_version": version,
        "chunks": chunks if chunks is not None else [chunk()],
    }


PROBE_RESULTS = {
    "P1": "PASS — nonempty whitespace-only chunks are accepted when needed to preserve a meaningful source block within the size limit; empty source blocks are filtered before chunk construction.",
    "P2": "PASS — zero chunks are accepted; useful for a valid empty chunk result.",
    "P3": "PASS — chunk text is not checked against source slices, so synthetic separators are accepted.",
    "P4": "PASS — leading and trailing whitespace around nonblank text are accepted.",
    "P5": "PASS — proposed `5c-1.0.0+0123456789ab` policy version is accepted.",
    "P6": "PASS — one block can contribute separate valid ranges in non-adjacent chunks.",
    "P7": "PASS — source ranges from multiple pages are permitted in one chunk.",
    "P8": "PASS — partial ranges over TABLE blocks are accepted.",
}


def test_p1_whitespace_only_chunk_is_accepted_but_empty_string_is_rejected():
    assert Chunk.model_validate(chunk(text=" \t\n")).text == " \t\n"
    with pytest.raises(ValidationError, match="Chunk text cannot be empty"):
        Chunk.model_validate(chunk(text=""))


def test_p2_zero_chunk_document_is_accepted():
    assert ChunkedDocument.model_validate(document(chunks=[])).chunks == ()


def test_p3_synthetic_separator_is_accepted():
    value = Chunk.model_validate(
        chunk(
            text="Evidence\n\nMore evidence",
            sources=[
                source(order=0, length=8, start=0, end=8),
                source(order=1, length=13, start=0, end=13),
            ],
        )
    )

    assert value.text == "Evidence\n\nMore evidence"
    assert len(value.sources) == 2


def test_p4_leading_and_trailing_whitespace_are_accepted():
    assert Chunk.model_validate(chunk(text="  Evidence  ")).text == "  Evidence  "


def test_p5_chunk_v1_policy_version_is_accepted():
    proposed_version = "5c-1.0.0+0123456789ab"
    assert ChunkedDocument.model_validate(document(version=proposed_version)).policy_version == proposed_version


def test_p6_block_can_recur_in_nonadjacent_chunks():
    result = ChunkedDocument.model_validate(document(chunks=[
        chunk(index=0, sources=[source(start=0, end=5)]),
        chunk(index=1, sources=[source(order=1)]),
        chunk(index=2, sources=[source(start=10, end=20)]),
    ]))
    assert [c.sources[0].block_id for c in result.chunks] == ["page-1-block-0", "page-1-block-1", "page-1-block-0"]


def test_p7_chunk_can_have_sources_from_multiple_pages():
    value = Chunk.model_validate(chunk(sources=[source(), source(page=2, order=0)]))
    assert {item.page_number for item in value.sources} == {1, 2}


def test_p8_partial_table_range_is_accepted():
    value = Chunk.model_validate(chunk(sources=[source(kind="TABLE", length=20, start=3, end=8)]))
    assert (value.sources[0].start_offset, value.sources[0].end_offset) == (3, 8)
