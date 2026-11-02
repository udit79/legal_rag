from pathlib import Path
from types import SimpleNamespace
from collections.abc import Iterable

import numpy
import pymupdf
import pytest

from src.ingestion.contracts import (
    DocumentRoutePlan,
    PageRouteDecision,
    Route,
)
from src.ingestion.dispatcher import dispatch_document
from src.ingestion.parser_contracts import (
    CanonicalBlock,
    CanonicalPage,
    SourceCoordinates,
)
from src.ingestion.parsers import (
    DoclingDocument,
    DoclingResult,
    FastTextParser,
    OcrParser,
    StructuredParser,
)

CORPUS = Path(__file__).parent / "fixtures" / "ingestion_corpus"


def make_plan(*routes: Route) -> DocumentRoutePlan:
    decisions = tuple(
        PageRouteDecision(
            page_number=page_number,
            route=route,
            reason="Test dispatch",
            decision_source="RULES",
            policy_version="test-v1",
        )
        for page_number, route in enumerate(routes, start=1)
    )
    all_blank = all(route == "SKIP_BLANK" for route in routes)
    return DocumentRoutePlan(
        page_count=len(routes),
        decisions=decisions,
        document_disposition="REJECT_NO_CONTENT" if all_blank else "PROCESS",
    )


def make_pdf(path: Path, page_count: int = 1) -> Path:
    document = pymupdf.open()
    for _ in range(page_count):
        page = document.new_page()
        page.insert_text((72, 72), "First line")
        page.insert_text((72, 96), "Second line")
    document.save(path)
    document.close()
    return path


def test_fast_text_parser_preserves_native_text_and_coordinates(tmp_path: Path):
    source = make_pdf(tmp_path / "native.pdf")

    page = FastTextParser().parse_page(source, 1)

    assert page.status == "PARSED"
    text_blocks = [block for block in page.blocks if block.block_type == "TEXT"]
    assert len(text_blocks) == 2
    assert "First line" in text_blocks[0].text
    assert "Second line" in text_blocks[1].text
    assert [block.reading_order for block in text_blocks] == [0, 1]
    assert all(block.source_coordinates is not None for block in text_blocks)
    assert all(
        block.source_coordinates.unit == "PDF_POINT"
        and block.source_coordinates.origin == "TOP_LEFT"
        for block in text_blocks
        if block.source_coordinates is not None
    )


def test_fast_text_parser_distinguishes_blank_and_vector_only_pages(tmp_path: Path):
    blank_source = tmp_path / "blank.pdf"
    blank_document = pymupdf.open()
    blank_document.new_page()
    blank_document.save(blank_source)
    blank_document.close()

    blank = FastTextParser().parse_page(blank_source, 1)
    assert blank.status == "BLANK"
    assert not blank.blocks

    vector_source = tmp_path / "vector.pdf"
    vector_document = pymupdf.open()
    vector_document.new_page().draw_rect(pymupdf.Rect(72, 72, 150, 120))
    vector_document.save(vector_source)
    vector_document.close()

    vector = FastTextParser().parse_page(vector_source, 1)
    assert vector.status == "VECTOR_ONLY"
    assert vector.blocks
    assert all(block.block_type == "VECTOR" for block in vector.blocks)


def test_structured_parser_returns_table_blocks_with_original_page_numbers(
    tmp_path: Path,
):
    source = make_pdf(tmp_path / "structured.pdf")
    bbox = SimpleNamespace(l=10.0, t=90.0, r=300.0, b=70.0)
    provenance = SimpleNamespace(page_no=1, bbox=bbox)

    class Table:
        label = SimpleNamespace(value="table")
        text = None
        prov = [provenance]

        def export_to_markdown(self, *, doc):
            assert doc is not None
            return "| Item |\n| --- |\n| Contract review |"

    class FakeDocument:
        def iterate_items(self) -> Iterable[tuple[object, object]]:
            yield Table(), 0

    class FakeResult:
        status: object
        errors: object
        document: DoclingDocument

        def __init__(self) -> None:
            self.status = SimpleNamespace(name="SUCCESS")
            self.errors = []
            self.document = FakeDocument()

    class FakeConverter:
        def convert(
            self,
            source: Path,
            *,
            page_range: tuple[int, int],
            raises_on_error: bool,
        ) -> DoclingResult:
            assert source.is_file()
            assert page_range == (1, 1)
            assert not raises_on_error
            return FakeResult()

    page = StructuredParser(converter=FakeConverter()).parse_page(source, 1)

    tables = [block for block in page.blocks if block.block_type == "TABLE"]
    assert page.page_number == 1
    assert tables
    assert "Contract review" in tables[0].text
    assert tables[0].source_coordinates is not None
    assert tables[0].source_coordinates.origin == "BOTTOM_LEFT"


def test_ocr_parser_uses_rapidocr_output_and_returns_canonical_blocks(tmp_path: Path):
    source = make_pdf(tmp_path / "ocr.pdf")

    class FakeRapidOCR:
        def __call__(self, image_path: str):
            assert Path(image_path).is_file()
            return SimpleNamespace(
                boxes=numpy.array([[[30, 40], [130, 40], [130, 60], [30, 60]]]),
                txts=("recognized",),
                scores=(0.95,),
            )

    page = OcrParser(
        engine="rapidocr",
        rapidocr_engine=FakeRapidOCR(),
    ).parse_page(source, 1)

    ocr_blocks = [block for block in page.blocks if block.text == "recognized"]
    assert page.status == "PARSED"
    assert len(ocr_blocks) == 1
    assert ocr_blocks[0].confidence == pytest.approx(95.0)
    assert ocr_blocks[0].source_coordinates is not None
    assert ocr_blocks[0].source_coordinates.unit == "PDF_POINT"


def test_ocr_parser_parses_tesseract_tsv(monkeypatch, tmp_path: Path):
    source = make_pdf(tmp_path / "tesseract.pdf")

    def fake_run(*_args, **_kwargs):
        assert _args
        assert _kwargs["timeout"] > 0
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\t"
                "top\twidth\theight\tconf\ttext\n"
                "5\t1\t1\t1\t1\t1\t10\t20\t40\t12\t92.5\tOCR\n"
                "5\t1\t1\t1\t1\t2\t55\t20\t60\t12\t90.5\ttext\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("src.ingestion.parsers.subprocess.run", fake_run)
    page = OcrParser(
        engine="tesseract", tesseract_command="tesseract-test"
    ).parse_page(source, 1)

    ocr_blocks = [block for block in page.blocks if block.text == "OCR text"]
    assert len(ocr_blocks) == 1
    assert ocr_blocks[0].confidence == pytest.approx(91.5)
    assert ocr_blocks[0].source_coordinates is not None
    assert ocr_blocks[0].source_coordinates.x0 == pytest.approx(2.4, abs=0.01)


class FakeParser:
    def __init__(self, parser_name: str, *, fail_page: int | None = None):
        self.parser_name = parser_name
        self.fail_page = fail_page
        self.pages: list[int] = []

    def parse_page(self, source: Path, page_number: int) -> CanonicalPage:
        assert source.is_file()
        self.pages.append(page_number)
        if page_number == self.fail_page:
            raise RuntimeError(f"failure on page {page_number}")
        return CanonicalPage(
            page_number=page_number,
            parser=self.parser_name,
            status="PARSED",
            blocks=(
                CanonicalBlock(
                    page_number=page_number,
                    text=f"page {page_number}",
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
                ),
            ),
        )


def test_dispatcher_selects_adapters_and_preserves_original_page_order(
    tmp_path: Path,
):
    source = make_pdf(tmp_path / "mixed.pdf", page_count=4)
    fast = FakeParser("FAST_TEXT")
    structured = FakeParser("DOCLING")
    ocr = FakeParser("TESSERACT")
    plan = make_plan("FAST_TEXT", "STRUCTURED_PARSE", "OCR", "LAYOUT_OCR")

    result = dispatch_document(
        source,
        plan,
        fast_text_parser=fast,
        structured_parser=structured,
        ocr_parser=ocr,
    )

    assert fast.pages == [1]
    assert structured.pages == [2, 4]
    assert ocr.pages == [3]
    assert [page.page_number for page in result.pages] == [1, 2, 3, 4]
    assert [page.parser for page in result.pages] == [
        "FAST_TEXT",
        "DOCLING",
        "TESSERACT",
        "DOCLING",
    ]


def test_dispatcher_records_page_failure_without_dropping_later_pages(
    tmp_path: Path,
):
    source = make_pdf(tmp_path / "partial.pdf", page_count=3)
    parser = FakeParser("FAST_TEXT", fail_page=2)

    result = dispatch_document(
        source,
        make_plan("FAST_TEXT", "FAST_TEXT", "FAST_TEXT"),
        fast_text_parser=parser,
    )

    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert result.pages[0].status == "PARSED"
    assert result.pages[1].status == "FAILED"
    assert result.pages[1].failure is not None
    assert result.pages[1].failure.code == "RuntimeError"
    assert result.pages[2].status == "PARSED"


def test_dispatcher_detects_corrupt_page_even_if_route_marks_it_blank():
    source = CORPUS / "pdfs" / "10_corrupted" / "10h_corrupt_page2_stream.pdf"

    result = dispatch_document(
        source,
        make_plan("FAST_TEXT", "SKIP_BLANK", "FAST_TEXT"),
    )

    assert result.pages[0].status == "PARSED"
    assert result.pages[1].status == "FAILED"
    assert result.pages[2].status == "PARSED"


def test_dispatcher_does_not_skip_content_when_route_marks_page_blank(
    tmp_path: Path,
):
    source = make_pdf(tmp_path / "misrouted.pdf")

    result = dispatch_document(source, make_plan("SKIP_BLANK"))

    assert result.document_disposition == "PROCESS"
    assert result.pages[0].status == "PARSED"
    assert result.pages[0].warnings[0].code == "ROUTE_FEATURE_MISMATCH"


def test_dispatcher_returns_valid_blank_page_when_skip_route_is_correct(
    tmp_path: Path,
):
    source = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(source)
    document.close()

    result = dispatch_document(source, make_plan("SKIP_BLANK"))

    assert result.document_disposition == "REJECT_NO_CONTENT"
    assert result.pages[0].status == "BLANK"


def test_dispatcher_records_corrupt_page_and_parses_other_pages():
    source = CORPUS / "pdfs" / "10_corrupted" / "10h_corrupt_page2_stream.pdf"

    result = dispatch_document(
        source,
        make_plan("FAST_TEXT", "FAST_TEXT", "FAST_TEXT"),
    )

    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert result.pages[0].status == "PARSED"
    assert result.pages[1].status == "FAILED"
    assert result.pages[2].status == "PARSED"


def test_dispatcher_records_missing_ocr_executable_and_continues(
    tmp_path: Path,
):
    source = make_pdf(tmp_path / "ocr-failure.pdf", page_count=2)

    result = dispatch_document(
        source,
        make_plan("OCR", "FAST_TEXT"),
        ocr_parser=OcrParser(
            engine="tesseract",
            tesseract_command=str(tmp_path / "missing-tesseract.exe"),
        ),
    )

    assert result.pages[0].status == "FAILED"
    assert result.pages[0].failure is not None
    assert result.pages[0].failure.code == "FileNotFoundError"
    assert result.pages[1].status == "PARSED"


def test_dispatcher_marks_encrypted_document_failure():
    source = CORPUS / "pdfs" / "10_corrupted" / "10g_encrypted_password_required.pdf"

    result = dispatch_document(source, make_plan("FAST_TEXT", "FAST_TEXT"))

    assert result.document_failure is not None
    assert result.document_failure.code == "ENCRYPTED_DOCUMENT"
    assert all(page.status == "FAILED" for page in result.pages)
