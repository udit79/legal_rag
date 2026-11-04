from src.ingestion.parser_contracts import (
    CanonicalBlock,
    SourceCoordinates,
)
from src.ingestion.parsers import _repair_multi_appeal_cover_order


def _block(order: int, text: str, x: float, y: float) -> CanonicalBlock:
    return CanonicalBlock(
        page_number=1,
        text=text,
        block_type="TEXT",
        reading_order=order,
        source_coordinates=SourceCoordinates(
            x0=x,
            y0=y,
            x1=x + 90,
            y1=y + 10,
            unit="PDF_POINT",
            origin="BOTTOM_LEFT",
        ),
    )


def _cover_blocks():
    # Input order deliberately interleaves the left and right party columns.
    return (
        _block(0, "CIVIL APPEAL NO. 1", 20, 90),
        _block(1, ".... RESPONDENT(S)", 300, 40),
        _block(2, "VERSUS", 150, 60),
        _block(3, ".... APPELLANT(S)", 300, 80),
        _block(4, "CIVIL APPEAL NO. 2", 20, 20),
        _block(5, "VERSUS", 150, 10),
        _block(6, ".... APPELLANT(S)", 300, 20),
        _block(7, ".... RESPONDENT(S)", 300, 0),
        _block(8, "IN THE SUPREME COURT OF INDIA", 20, 120),
    )


def test_multi_appeal_cover_orders_visual_rows_and_columns_preserving_blocks():
    original = _cover_blocks()

    repaired = _repair_multi_appeal_cover_order(original)

    assert [block.text for block in repaired] == [
        "IN THE SUPREME COURT OF INDIA",
        "CIVIL APPEAL NO. 1",
        ".... APPELLANT(S)",
        "VERSUS",
        ".... RESPONDENT(S)",
        "CIVIL APPEAL NO. 2",
        ".... APPELLANT(S)",
        "VERSUS",
        ".... RESPONDENT(S)",
    ]
    assert [block.reading_order for block in repaired] == list(range(len(original)))
    assert {block.block_id for block in repaired} == {block.block_id for block in original}
    assert {block.text: block.source_coordinates for block in repaired} == {
        block.text: block.source_coordinates for block in original
    }


def test_ambiguous_page_falls_back_to_docling_order():
    original = _cover_blocks()[:-1]

    assert _repair_multi_appeal_cover_order(original) == original


def test_cover_page_repair_preserves_block_id_when_reading_order_changes():
    original = _cover_blocks()
    repaired = _repair_multi_appeal_cover_order(original)
    original_by_id = {block.block_id: block for block in original}

    for block in repaired:
        assert block.block_id in original_by_id
        assert block.source_coordinates == original_by_id[block.block_id].source_coordinates
