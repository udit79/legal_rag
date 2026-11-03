import pytest
from pydantic import ValidationError

from src.ingestion.chunking import (
    ChunkingConfig,
    ChunkingPreparation,
    PreparedPage,
    prepare_document_for_chunking,
)
from src.ingestion.parser_contracts import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalPage,
    ParserFailure,
    SourceCoordinates,
)


def block(page, order, kind, text):
    return CanonicalBlock(
        page_number=page,
        text=text,
        block_type=kind,
        reading_order=order,
        source_coordinates=SourceCoordinates(
            x0=0, y0=0, x1=10, y1=10, unit="PDF_POINT", origin="TOP_LEFT"
        ),
    )


def parsed_page(*blocks, number=1, warnings=()):
    return CanonicalPage(
        page_number=number,
        parser="test",
        status="PARSED",
        blocks=tuple(blocks),
        warnings=warnings,
    )


def test_all_block_types_follow_text_eligibility_rules_and_preserve_source_text():
    page = parsed_page(
        block(1, 0, "TEXT", "  Text\n"),
        block(1, 1, "TABLE", "row 1\nrow 2"),
        block(1, 2, "OTHER", "Other content"),
        block(1, 3, "IMAGE", "OCR text"),
        block(1, 4, "VECTOR", "Vector label"),
    )

    result = prepare_document_for_chunking((page,))

    assert isinstance(result, ChunkingPreparation)
    assert [item.text for item in result.eligible_blocks] == [
        "  Text\n",
        "row 1\nrow 2",
        "Other content",
        "OCR text",
    ]
    assert result.pages[0].outcome == "READY_FOR_SEGMENTATION"
    assert [(item.block_type, item.reason) for item in result.skipped_blocks] == [
        ("VECTOR", "VECTOR_NOT_TEXT_ELIGIBLE")
    ]


def test_image_text_eligibility_respects_config():
    page = parsed_page(block(1, 0, "IMAGE", "OCR text"))

    included = prepare_document_for_chunking((page,), ChunkingConfig())
    excluded = prepare_document_for_chunking(
        (page,), ChunkingConfig(include_image_text=False)
    )

    assert included.pages[0].outcome == "READY_FOR_SEGMENTATION"
    assert included.eligible_blocks == page.blocks
    assert excluded.pages[0].outcome == "SKIPPED_NO_ELIGIBLE_TEXT"
    assert excluded.eligible_blocks == ()
    assert excluded.skipped_blocks[0].reason == "IMAGE_TEXT_DISABLED"


@pytest.mark.parametrize(
    ("kind", "text", "expected_reason"),
    [
        ("TEXT", "", "EMPTY_TEXT"),
        ("TEXT", " \t\n", "WHITESPACE_ONLY_TEXT"),
        ("TABLE", "", "EMPTY_TEXT"),
        ("TABLE", " \t\n", "WHITESPACE_ONLY_TEXT"),
        ("OTHER", "", "EMPTY_TEXT"),
        ("OTHER", " \t\n", "WHITESPACE_ONLY_TEXT"),
        ("IMAGE", "", "IMAGE_NO_USABLE_TEXT"),
        ("IMAGE", " \t\n", "IMAGE_NO_USABLE_TEXT"),
        ("VECTOR", "", "VECTOR_NOT_TEXT_ELIGIBLE"),
    ],
)
def test_empty_or_whitespace_blocks_are_recorded_not_eligible(
    kind, text, expected_reason
):
    page = parsed_page(block(1, 0, kind, text))

    result = prepare_document_for_chunking((page,))

    assert result.eligible_blocks == ()
    assert len(result.skipped_blocks) == 1
    assert result.skipped_blocks[0].reason == expected_reason
    expected_page_outcome = (
        "SKIPPED_IMAGE_NO_TEXT" if kind == "IMAGE" else "SKIPPED_NO_ELIGIBLE_TEXT"
    )
    assert result.pages[0].outcome == expected_page_outcome


def test_page_routes_cover_each_parser_status_and_image_only_case():
    warning = ParserFailure(parser="test", code="warning", message="Parser warning")
    failure = ParserFailure(parser="test", code="failed", message="Parser failed")
    pages = (
        parsed_page(block(1, 0, "TEXT", "eligible"), number=1),
        parsed_page(block(2, 0, "IMAGE", ""), number=2),
        CanonicalPage(page_number=3, parser="test", status="BLANK"),
        CanonicalPage(
            page_number=4,
            parser="test",
            status="VECTOR_ONLY",
            blocks=(block(4, 0, "VECTOR", ""),),
        ),
        CanonicalPage(
            page_number=5,
            parser="test",
            status="FAILED",
            failure=failure,
            warnings=(warning,),
        ),
        parsed_page(block(6, 0, "TEXT", "\n "), number=6, warnings=(warning,)),
    )

    result = prepare_document_for_chunking(pages)

    assert [page.outcome for page in result.pages] == [
        "READY_FOR_SEGMENTATION",
        "SKIPPED_IMAGE_NO_TEXT",
        "SKIPPED_BLANK",
        "SKIPPED_VECTOR_ONLY",
        "SKIPPED_FAILED",
        "SKIPPED_NO_ELIGIBLE_TEXT",
    ]
    assert result.pages[0].eligible_blocks[0].text == "eligible"
    assert result.pages[4].failure == failure
    assert result.pages[4].warnings == (warning,)
    assert result.pages[5].warnings == (warning,)


def test_mixed_pages_continue_after_failure_and_empty_inputs_are_valid():
    failure = ParserFailure(parser="test", code="failed", message="Parser failed")
    document = CanonicalDocument(
        page_count=3,
        document_disposition="PROCESS",
        pages=(
            parsed_page(block(1, 0, "TEXT", "keep going"), number=1),
            CanonicalPage(
                page_number=2, parser="test", status="FAILED", failure=failure
            ),
            CanonicalPage(page_number=3, parser="test", status="BLANK"),
        ),
    )

    result = prepare_document_for_chunking(document)
    empty = prepare_document_for_chunking(())

    assert [page.outcome for page in result.pages] == [
        "READY_FOR_SEGMENTATION",
        "SKIPPED_FAILED",
        "SKIPPED_BLANK",
    ]
    assert [block.text for block in result.eligible_blocks] == ["keep going"]
    assert result.pages[1].failure == failure
    assert empty.pages == ()
    assert empty.eligible_blocks == ()
    assert empty.skipped_blocks == ()


def test_failed_canonical_page_rejects_partial_text_blocks():
    """FAILED pages cannot retain partial blocks under CanonicalPage's contract."""
    failure = ParserFailure(parser="test", code="failed", message="Parser failed")

    with pytest.raises(ValidationError, match="Failed pages require a failure"):
        CanonicalPage(
            page_number=1,
            parser="test",
            status="FAILED",
            blocks=(block(1, 0, "TEXT", "partial extracted text"),),
            failure=failure,
        )
