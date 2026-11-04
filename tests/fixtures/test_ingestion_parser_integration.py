"""Integration tests for the selected PDF ingestion parser stack.

Selected stack under test
-------------------------
* PyMuPDF: PDF opening, page-level native text, image/drawing inspection, rendering.
* Docling: structured layout, reading order and table structure.
* Tesseract 5+: standalone OCR baseline for rendered scanned/vector-text pages.
* Docling + RapidOCR: optional neural OCR comparison through Docling's OCR options.

Expected corpus layout:
    tests/fixtures/ingestion_corpus/manifest.json
    tests/fixtures/ingestion_corpus/pdfs/...
    tests/fixtures/ingestion_corpus/images/...
    tests/fixtures/ingestion_corpus/expected/...

Run from repository root:
    uv run python -m pytest tests/fixtures/test_ingestion_parser_integration.py -v

Tesseract must be installed as a Windows executable. Set TESSERACT_CMD if it is
not available on PATH. Docling's first conversion may download model artifacts.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytest
from PIL import Image

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.ingestion.parsers import _docling_deprecation_compat


def _find_corpus_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = (
        here / "fixtures" / "ingestion_corpus",  # tests/test_*.py
        here / "ingestion_corpus",  # tests/fixtures/test_*.py
        here / "tests" / "fixtures" / "ingestion_corpus",  # repository root
        Path.cwd() / "tests" / "fixtures" / "ingestion_corpus",
    )
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate ingestion_corpus/manifest.json. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


CORPUS = _find_corpus_dir()
MANIFEST: dict[str, Any] = json.loads(
    (CORPUS / "manifest.json").read_text(encoding="utf-8")
)
DOCS: dict[int, dict[str, Any]] = {
    doc["case_no"]: doc for doc in MANIFEST["documents"]
}


def norm(value: str) -> str:
    return " ".join(value.split())


def compact(value: str) -> str:
    """Remove punctuation/whitespace for OCR markers containing numbers."""
    return re.sub(r"[^A-Za-z0-9]", "", value).casefold()


def expected_blocks(relative_path: str) -> list[str]:
    path = CORPUS / relative_path
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pdf_path(case_no: int) -> Path:
    return CORPUS / DOCS[case_no]["file"]


def pymupdf_text(page: pymupdf.Page) -> str:
    value: object = page.get_text("text", sort=False)
    if not isinstance(value, str):
        raise TypeError(
            "PyMuPDF get_text('text') returned "
            f"{type(value).__name__}, expected str"
        )
    return value


def pymupdf_body_text_excluding_rotated_spans(page: pymupdf.Page) -> str:
    """Extract horizontal text spans, excluding rotated watermark spans.

    This is a fixture-specific baseline helper, not a general-purpose watermark
    detector. A production normalizer should make this decision from layout and
    document policy rather than discarding every rotated span unconditionally.
    """
    page_dict: Any = page.get_text("dict", sort=False)
    pieces: list[str] = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                direction = span.get("dir", (1.0, 0.0))
                if len(direction) >= 2 and (abs(direction[1]) > 0.02 or abs(direction[0]) < 0.98):
                    continue
                text = span.get("text", "")
                if isinstance(text, str) and text.strip():
                    pieces.append(text)
    return " ".join(pieces)


def clause_ids(text: str) -> list[str]:
    return re.findall(r"(?<!\d)([78]\.\d)(?!\d)", text)


def _tesseract_command() -> str:
    configured = os.environ.get("TESSERACT_CMD")
    command = configured or shutil.which("tesseract")
    if not command:
        pytest.fail(
            "Tesseract executable was not found. Install Tesseract OCR, add "
            "tesseract.exe to PATH, or set TESSERACT_CMD to its full path.",
            pytrace=False,
        )
    return command


def _ocr_pdf_page(pdf: Path, page_number: int, tmp_path: Path) -> str:
    """Render a one-based page with PyMuPDF and run Tesseract CLI on the raster."""
    command = _tesseract_command()
    image_path = tmp_path / f"{pdf.stem}-page-{page_number}.png"
    with pymupdf.open(pdf) as document:
        if document.needs_pass:
            raise RuntimeError(f"PDF requires a password: {pdf.name}")
        if page_number < 1 or page_number > len(document):
            raise ValueError(f"Invalid page number {page_number} for {pdf.name}")
        pixmap = document[page_number - 1].get_pixmap(
            dpi=300, colorspace=pymupdf.csRGB, alpha=False
        )
        pixmap.save(image_path)

    result = subprocess.run(
        [command, str(image_path), "stdout", "-l", "eng", "--oem", "1", "--psm", "6"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Tesseract failed for {pdf.name} page {page_number}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def _assert_docling_success(result: Any, source: Path) -> Any:
    status = getattr(result, "status", "UNKNOWN")
    status_name = str(getattr(status, "name", getattr(status, "value", status))).upper()
    assert status_name == "SUCCESS", (
        f"Docling did not fully convert {source.name}: status={status_name}; "
        f"errors={getattr(result, 'errors', None)}"
    )
    return result.document


def _docling_markdown(document: Any) -> str:
    markdown = document.export_to_markdown()
    assert isinstance(markdown, str), (
        "Docling export_to_markdown() must return a string; "
        f"got {type(markdown).__name__}"
    )
    return markdown


def _table_shape(table: Any) -> tuple[int, int]:
    data = getattr(table, "data", None)
    rows = getattr(data, "num_rows", None)
    columns = getattr(data, "num_cols", None)
    if not isinstance(rows, int) or not isinstance(columns, int):
        raise AssertionError("Docling table is missing integer data.num_rows/num_cols")
    return rows, columns


def _table_page_number(table: Any) -> int | None:
    provenance = getattr(table, "prov", None)
    if not provenance:
        return None
    page_number = getattr(provenance[0], "page_no", None)
    return page_number if isinstance(page_number, int) else None


def _expected_csv(relative_path: str) -> list[list[str]]:
    with (CORPUS / relative_path).open(encoding="utf-8", newline="") as handle:
        return [[norm(cell) for cell in row] for row in csv.reader(handle)]


@pytest.fixture(scope="session")
def docling_layout_converter() -> DocumentConverter:
    """Docling structured layout/table parsing without running its OCR engine."""
    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        generate_page_images=True,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


@pytest.fixture(scope="session")
def docling_rapidocr_converter() -> DocumentConverter:
    """Docling configured explicitly to use RapidOCR for OCR comparison."""
    options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=RapidOcrOptions(),
        do_table_structure=True,
        generate_page_images=True,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


# ---------------------------------------------------------------------------
# PyMuPDF: native text and PDF primitives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("case_no", "page_number"),
    [
        (1, 1), (1, 2),
        (3, 1), (3, 3),
        (4, 1), (4, 2),
        (5, 1), (5, 2),
        (8, 1), (8, 2),
        (9, 1), (9, 3), (9, 5), (9, 7),
    ],
)
def test_pymupdf_native_text_contains_ground_truth(
    case_no: int, page_number: int
) -> None:
    spec = DOCS[case_no]["pages"][page_number - 1]
    assert spec["text_layer"] is True
    assert spec.get("ground_truth"), (
        f"Missing ground truth for {DOCS[case_no]['id']} page {page_number}"
    )

    with pymupdf.open(pdf_path(case_no)) as document:
        assert len(document) == DOCS[case_no]["page_count"]
        page = document[page_number - 1]
        # The watermark fixture deliberately contains a diagonal text run; exclude
        # rotated spans from body-text completeness checks for that page.
        if case_no == 8 and page_number == 1:
            text = norm(pymupdf_body_text_excluding_rotated_spans(page))
        else:
            text = norm(pymupdf_text(page))

    for block in expected_blocks(spec["ground_truth"]):
        assert norm(block) in text, (
            f"PyMuPDF missed ground-truth text in {DOCS[case_no]['id']} "
            f"page {page_number}: {block[:80]!r}"
        )


def test_pymupdf_preserves_multicolumn_clause_order_in_content_stream() -> None:
    """The synthetic fixture deliberately writes text in reading order."""
    with pymupdf.open(pdf_path(5)) as document:
        for page_index in range(len(document)):
            spec = DOCS[5]["pages"][page_index]
            text = pymupdf_text(document[page_index])
            expected = [
                value for value in spec["reading_order"]
                if re.fullmatch(r"\d\.\d", value)
            ]
            assert clause_ids(text) == expected


def test_pymupdf_detects_vector_only_pages_without_treating_them_as_blank() -> None:
    with pymupdf.open(pdf_path(6)) as document:
        for page in document:
            assert not pymupdf_text(page).strip()
            assert len(page.get_images(full=True)) == 0
            assert len(page.get_drawings()) > 0
            pixmap = page.get_pixmap(
                dpi=72, colorspace=pymupdf.csGRAY, alpha=False
            )
            pixels = np.frombuffer(pixmap.samples, dtype=np.uint8)
            assert pixels.size > 0
            assert float((pixels < 245).mean()) > 0.001


def test_pymupdf_distinguishes_truly_blank_page_from_vector_page() -> None:
    with pymupdf.open(pdf_path(7)) as document:
        page = document[0]
        assert not pymupdf_text(page).strip()
        assert len(page.get_images(full=True)) == 0
        assert len(page.get_drawings()) == 0
        pixmap = page.get_pixmap(
            dpi=72, colorspace=pymupdf.csGRAY, alpha=False
        )
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8)
        assert float((pixels < 128).mean()) == 0.0


def test_pymupdf_detects_encrypted_pdf_as_password_required() -> None:
    encrypted_pdf = CORPUS / DOCS[10]["members"][6]["file"]
    with pymupdf.open(encrypted_pdf) as document:
        assert bool(document.needs_pass)


def test_pymupdf_extracts_rotated_watermark_without_losing_body_ground_truth() -> None:
    spec = DOCS[8]["pages"][0]
    with pymupdf.open(pdf_path(8)) as document:
        text = norm(pymupdf_text(document[0]))
    for block in expected_blocks(spec["ground_truth"]):
        assert norm(block) in text, f"Missing body text under watermark: {block[:60]!r}"
    assert "DRAFT - NOT FOR EXECUTION" in text


# ---------------------------------------------------------------------------
# Docling: structured layout, tables and reading order
# ---------------------------------------------------------------------------
def test_docling_converts_ruled_and_borderless_tables(
    docling_layout_converter: DocumentConverter,
) -> None:
    source = pdf_path(4)
    with _docling_deprecation_compat():
        result = docling_layout_converter.convert(source)
    doc = _assert_docling_success(result, source)
    tables = list(doc.tables)
    shapes_by_page: dict[int | None, list[tuple[int, int]]] = {}
    for table in tables:
        shapes_by_page.setdefault(_table_page_number(table), []).append(_table_shape(table))

    assert (7, 4) in shapes_by_page.get(1, []), (
        f"Docling did not recover the ruled 7x4 table; found {shapes_by_page}"
    )
    assert (4, 3) in shapes_by_page.get(2, []), (
        f"Docling did not recover the borderless 4x3 table; found {shapes_by_page}"
    )

    markdown = norm(_docling_markdown(doc))
    for page_spec in DOCS[4]["pages"]:
        expected_rows = _expected_csv(page_spec["table_csv"])
        for row in expected_rows:
            for cell in row:
                if cell:
                    assert cell in markdown, (
                        f"Docling Markdown is missing expected table cell {cell!r}"
                    )


def test_docling_preserves_multicolumn_clause_order(
    docling_layout_converter: DocumentConverter,
) -> None:
    source = pdf_path(5)
    with _docling_deprecation_compat():
        converted = docling_layout_converter.convert(source)
    doc = _assert_docling_success(converted, source)
    markdown = _docling_markdown(doc)
    expected: list[str] = []
    for page_spec in DOCS[5]["pages"]:
        expected.extend(
            value for value in page_spec["reading_order"]
            if re.fullmatch(r"\d\.\d", value)
        )
    actual = clause_ids(markdown)
    assert actual == expected, (
        "Docling reading order differs from manifest. "
        f"Expected {expected}; got {actual}"
    )


def test_docling_extracts_watermarked_document_body(
    docling_layout_converter: DocumentConverter,
) -> None:
    source = pdf_path(8)
    with _docling_deprecation_compat():
        converted = docling_layout_converter.convert(source)
    doc = _assert_docling_success(converted, source)
    markdown = norm(_docling_markdown(doc))
    for page_spec in DOCS[8]["pages"]:
        for block in expected_blocks(page_spec["ground_truth"]):
            if re.fullmatch(r"Page \d+ of \d+", block):
                continue
            assert norm(block) in markdown, (
                f"Docling did not recover watermarked document text: {block[:60]!r}"
            )


# ---------------------------------------------------------------------------
# Tesseract CLI: baseline OCR on rendered raster pages
# ---------------------------------------------------------------------------
def test_tesseract_ocr_recovers_scanned_invoice_markers(tmp_path: Path) -> None:
    spec = DOCS[2]["pages"][0]
    actual = compact(_ocr_pdf_page(pdf_path(2), 1, tmp_path))
    for marker in spec["ocr_tokens_must_contain"]:
        assert compact(marker) in actual, (
            f"Tesseract failed to recover marker {marker!r} from case 02"
        )


def test_tesseract_ocr_recovers_scanned_pages_in_mixed_pdf(tmp_path: Path) -> None:
    source = pdf_path(3)
    for page_number in (2, 4):
        spec = DOCS[3]["pages"][page_number - 1]
        expected_text = " ".join(expected_blocks(spec["ground_truth"]))
        expected_tokens = {
            compact(token)
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9,./'-]*", expected_text)
            if len(compact(token)) >= 5
        }
        actual_tokens = {
            compact(token)
            for token in re.findall(
                r"[A-Za-z0-9][A-Za-z0-9,./'-]*", _ocr_pdf_page(source, page_number, tmp_path)
            )
            if len(compact(token)) >= 5
        }
        assert expected_tokens, f"No usable OCR ground-truth tokens for page {page_number}"
        recall = len(expected_tokens & actual_tokens) / len(expected_tokens)
        assert recall >= 0.60, (
            f"Tesseract ground-truth token recall too low on case 03 page {page_number}: "
            f"{recall:.1%} (minimum 60%)"
        )


def test_tesseract_ocr_recovers_vector_outlined_text(tmp_path: Path) -> None:
    spec = DOCS[6]["pages"][1]
    actual = compact(_ocr_pdf_page(pdf_path(6), 2, tmp_path))
    expected = " ".join(expected_blocks(spec["ground_truth"]))
    # Outlined characters have no native text layer; OCR must operate on a rendered page.
    key_tokens = [
        compact(token) for token in re.findall(r"[A-Za-z0-9]+", expected)
        if len(compact(token)) >= 5
    ]
    assert key_tokens, "Vector-outlined-text fixture needs ground truth"
    matched = sum(token in actual for token in key_tokens)
    assert matched / len(key_tokens) >= 0.60, (
        "Tesseract failed to recover enough text from vector-outlined page 2: "
        f"{matched}/{len(key_tokens)} key tokens"
    )


# ---------------------------------------------------------------------------
# Docling + RapidOCR: explicit neural OCR comparison to Tesseract baseline
# ---------------------------------------------------------------------------
def test_docling_rapidocr_recovers_scanned_invoice_markers(
    docling_rapidocr_converter: DocumentConverter,
) -> None:
    source = pdf_path(2)
    with _docling_deprecation_compat():
        result = docling_rapidocr_converter.convert(source)
    doc = _assert_docling_success(result, source)
    actual = compact(_docling_markdown(doc))
    for marker in DOCS[2]["pages"][0]["ocr_tokens_must_contain"]:
        assert compact(marker) in actual, (
            f"Docling+RapidOCR failed to recover marker {marker!r} from case 02"
        )


def test_standalone_images_match_manifest_dimensions_and_blank_flags() -> None:
    for item in MANIFEST["images"]:
        image_path = CORPUS / item["file"]
        assert image_path.is_file(), item["file"]
        with Image.open(image_path) as image:
            assert image.size == (item["width"], item["height"]), item["file"]
            if item.get("is_blank"):
                grayscale = np.asarray(image.convert("L"))
                dark_ratio = float((grayscale < 128).mean())
                assert dark_ratio <= item["dark_pixel_ratio_max"], item["file"]
