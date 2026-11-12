from pathlib import Path

from src.ingestion.chunking import chunk_document
from src.ingestion.dispatcher import dispatch_document
from src.ingestion.parser_contracts import (
    CanonicalBlock,
    CanonicalPage,
    SourceCoordinates,
)
from src.ingestion.pdf_bullet_artifacts import normalize_pdf_bullet_artifacts
from src.ingestion.preflight import inspect_pdf
from src.ingestion.router import route_document
from src.security.ingestion_screening.enums import AdmissionDecision
from src.security.ingestion_screening.screening_service import ScreeningService


def _block(page: int, order: int, text: str, y: float) -> CanonicalBlock:
    return CanonicalBlock(
        page_number=page,
        reading_order=order,
        text=text,
        block_type="TEXT",
        source_coordinates=SourceCoordinates(
            x0=100,
            y0=y,
            x1=500,
            y1=y + 12,
            unit="PDF_POINT",
            origin="TOP_LEFT",
        ),
    )


def _page(*texts: str) -> CanonicalPage:
    blocks = tuple(_block(1, index, text, index * 20) for index, text in enumerate(texts))
    return CanonicalPage(page_number=1, parser="TEST", status="PARSED", blocks=blocks)


def test_security_report_page_9_bullets_are_visible_auditable_and_screen_clean():
    source = Path("test_data/security_report.pdf")
    features = inspect_pdf(source)
    plan = route_document(features, "test-policy")
    canonical = dispatch_document(source, plan)
    page = canonical.pages[8]
    transformed = [
        (block, edit)
        for block in page.blocks
        for edit in block.text_transformations
    ]

    assert len(transformed) == 4
    assert all(edit.original_character == "\x01" for _, edit in transformed)
    assert all(edit.replacement == "➢" for _, edit in transformed)
    assert all(edit.source_page == 9 for _, edit in transformed)
    assert all(block.raw_text[0] == "\x01" for block, _ in transformed)
    assert all(block.text[0] == "➢" for block, _ in transformed)

    chunks = chunk_document(canonical, "security-report-test").document.chunks
    page_chunks = [
        chunk
        for chunk in chunks
        if any(source.page_number == 9 for source in chunk.sources)
    ]
    recommendation_chunks = [chunk for chunk in page_chunks if "➢" in chunk.text]
    assert page_chunks
    assert len(recommendation_chunks) == 1
    assert recommendation_chunks[0].index == 21
    assert recommendation_chunks[0].text.count("➢") == 4
    assert all(
        ScreeningService().screen_extracted_text(chunk.text)[0]
        == AdmissionDecision.ADMITTED
        for chunk in page_chunks
    )


def test_single_bullet_is_validated_by_nearby_recommendations_heading():
    normalized = normalize_pdf_bullet_artifacts(
        _page("4.5\nRecommendations", "\x01\nThe department should review backups.")
    )
    bullet = normalized.blocks[1]

    assert bullet.text.startswith("➢\n")
    assert bullet.raw_text.startswith("\x01\n")
    assert len(bullet.text_transformations) == 1


def test_common_bullet_and_arrow_glyphs_normalize_only_in_validated_lists():
    for marker in "•●‣⁃→➔➜➤-–—*+":
        page = normalize_pdf_bullet_artifacts(
            _page("Recommendations", f"{marker} Review the backups.")
        )
        bullet = page.blocks[1]
        edit = bullet.text_transformations[0]

        assert bullet.text == "➢ Review the backups."
        assert bullet.raw_text == f"{marker} Review the backups."
        assert edit.original_character == marker
        assert edit.replacement == "➢"
        assert ScreeningService().screen_extracted_text(bullet.text)[0] == (
            AdmissionDecision.ADMITTED
        )

    ordinary = normalize_pdf_bullet_artifacts(_page("The arrow → points right."))
    assert ordinary.blocks[0].text == "The arrow → points right."
    assert ordinary.blocks[0].text_transformations == ()


def test_control_character_inside_ordinary_prose_is_not_normalized():
    page = normalize_pdf_bullet_artifacts(_page("The value \x01 appears in prose."))

    assert page.blocks[0].text == "The value \x01 appears in prose."
    assert page.blocks[0].text_transformations == ()
    decision, flags = ScreeningService().screen_extracted_text(page.blocks[0].text)
    assert decision == AdmissionDecision.ADMITTED_FLAGGED
    assert any(flag.code == "UNEXPECTED_CONTROL_CHARACTER" for flag in flags)


def test_multiple_controls_outside_list_positions_remain_quarantined():
    text = "\x01ordinary opening text and \x02another control"
    page = normalize_pdf_bullet_artifacts(_page(text))
    decision, flags = ScreeningService().screen_extracted_text(page.blocks[0].text)

    assert page.blocks[0].text == text
    assert decision == AdmissionDecision.QUARANTINED
    assert any(
        flag.code == "MULTIPLE_UNEXPECTED_CONTROL_CHARACTERS" for flag in flags
    )


def test_mixed_valid_bullet_and_suspicious_control_preserves_g1_detection():
    page = normalize_pdf_bullet_artifacts(
        _page("Recommendations", "\x01\nReview backups \x02regularly.")
    )
    text = page.blocks[1].text
    decision, flags = ScreeningService().screen_extracted_text(text)

    assert text == "➢\nReview backups \x02regularly."
    assert page.blocks[1].raw_text == "\x01\nReview backups \x02regularly."
    assert decision == AdmissionDecision.ADMITTED_FLAGGED
    assert any(flag.code == "UNEXPECTED_CONTROL_CHARACTER" for flag in flags)


def test_transformation_offsets_map_to_identical_raw_and_derived_positions():
    original = "\x01\nRecommendation text"
    block = normalize_pdf_bullet_artifacts(
        _page("Recommendations", original)
    ).blocks[1]
    edit = block.text_transformations[0]

    assert edit.start_offset == 0
    assert edit.end_offset == 1
    assert block.raw_text[edit.start_offset : edit.end_offset] == edit.original_character
    assert block.text[edit.start_offset : edit.end_offset] == edit.replacement
    assert len(block.raw_text) == len(block.text)
