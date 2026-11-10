from src.ingestion.chunking import ChunkingConfig, chunk_document
from src.ingestion.parser_contracts import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalPage,
    SourceCoordinates,
)


def test_retrieval_text_removes_repeated_boundaries_and_repairs_hyphens_reversibly():
    header = "IN THE SUPREME COURT OF INDIA"
    footer = "Recurring administrative footer text"
    pages = tuple(
        CanonicalPage(
            page_number=page_number,
            parser="TEST",
            status="PARSED",
            blocks=(
                CanonicalBlock(
                    page_number=page_number,
                    text=(
                        f"{header}\nThe arbitra-\ntion continues on page "
                        f"{page_number}.\n{footer}"
                    ),
                    block_type="TEXT",
                    reading_order=0,
                    source_coordinates=SourceCoordinates(
                        x0=0,
                        y0=0,
                        x1=300,
                        y1=500,
                        unit="PDF_POINT",
                        origin="TOP_LEFT",
                    ),
                ),
            ),
        )
        for page_number in range(1, 4)
    )
    document = CanonicalDocument(
        page_count=3,
        document_disposition="PROCESS",
        pages=pages,
    )

    chunking = chunk_document(
        document,
        "retrieval-normalization-test",
        ChunkingConfig(max_chunk_chars=256, overlap_chars=0),
    )

    assert len(chunking.document.chunks) == 3
    for chunk in chunking.document.chunks:
        assert chunk.retrieval_text is not None
        assert chunk.text.startswith(header)
        assert footer in chunk.text
        assert header not in chunk.retrieval_text
        assert footer not in chunk.retrieval_text
        assert "arbitra-\ntion" in chunk.text
        assert "arbitration" in chunk.retrieval_text
        start = chunk.retrieval_text.index("arbitration")
        mapped_ranges = chunk.raw_text_ranges_for_retrieval_range(
            start,
            start + len("arbitration"),
        )
        assert len(mapped_ranges) == 2
        assert "".join(chunk.text[left:right] for left, right in mapped_ranges) == (
            "arbitration"
        )
