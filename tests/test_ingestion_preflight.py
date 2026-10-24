import hashlib
from pathlib import Path

import pymupdf
import pytest

from src.ingestion.preflight import InvalidPdfError, PdfNotFoundError, inspect_pdf


def test_inspect_pdf_returns_hash_size_and_per_page_counts(tmp_path: Path):
    pdf_path = tmp_path / "two-pages.pdf"
    document = pymupdf.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "Hello")
    second_page = document.new_page()
    image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1, 1), 0)
    image.clear_with(255)
    image_bytes = image.tobytes("png")
    second_page.insert_image(
        pymupdf.Rect(72, 72, 100, 100),
        stream=image_bytes,
    )
    second_page.insert_image(
        pymupdf.Rect(110, 72, 138, 100),
        stream=image_bytes,
    )
    document.save(pdf_path)
    document.close()

    pdf_bytes = pdf_path.read_bytes()
    features = inspect_pdf(pdf_path)

    assert features.content_sha256 == hashlib.sha256(pdf_bytes).hexdigest()
    assert features.file_size_bytes == len(pdf_bytes)
    assert features.page_count == 2
    assert tuple(page.model_dump() for page in features.pages) == (
        {"page_number": 1, "text_char_count": 6, "image_count": 0, "drawing_count": 0},
        {"page_number": 2, "text_char_count": 0, "image_count": 2, "drawing_count": 0},
    )


def test_inspect_pdf_accepts_valid_page_with_zero_text_and_images(tmp_path: Path):
    pdf_path = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    features = inspect_pdf(pdf_path)

    assert features.page_count == 1
    assert features.pages[0].model_dump() == {
        "page_number": 1,
        "text_char_count": 0,
        "image_count": 0,
        "drawing_count": 0,
    }


def test_inspect_pdf_counts_vector_drawings_without_marking_page_blank(tmp_path: Path):
    pdf_path = tmp_path / "vector.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.draw_rect(pymupdf.Rect(72, 72, 150, 120))
    document.save(pdf_path)
    document.close()

    features = inspect_pdf(pdf_path)

    assert features.pages[0].text_char_count == 0
    assert features.pages[0].image_count == 0
    assert features.pages[0].drawing_count > 0


def test_inspect_pdf_raises_documented_error_for_missing_file(tmp_path: Path):
    pdf_path = tmp_path / "missing.pdf"

    with pytest.raises(PdfNotFoundError, match="PDF file not found"):
        inspect_pdf(pdf_path)


@pytest.mark.parametrize("contents", [b"", b"not a PDF"])
def test_inspect_pdf_raises_documented_error_for_invalid_pdf(
    tmp_path: Path, contents: bytes
):
    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_bytes(contents)

    with pytest.raises(InvalidPdfError, match="Unable to read PDF file"):
        inspect_pdf(pdf_path)
