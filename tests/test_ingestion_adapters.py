from pathlib import Path
from collections.abc import Iterable
from types import SimpleNamespace
from typing import TypeAlias, cast

import numpy
import pymupdf
import pytest
from docling_core.types.doc.base import CoordOrigin

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
    _native_text_line_geometry,
)

CORPUS = Path(__file__).parent / "fixtures" / "ingestion_corpus"
PdfBounds: TypeAlias = tuple[float, float, float, float]


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
        page.insert_textbox(
            pymupdf.Rect(72, 72, 400, 150),
            "First line\nSecond line",
        )
    document.save(path)
    document.close()
    return path


def assert_pdf_point_bounds(source: Path, page: CanonicalPage) -> None:
    with pymupdf.open(source) as document:
        page_rect = document[page.page_number - 1].rect

    for block in page.blocks:
        coordinates = block.source_coordinates
        assert coordinates.unit == "PDF_POINT"
        assert 0 <= coordinates.x0 <= coordinates.x1 <= page_rect.width
        assert 0 <= coordinates.y0 <= coordinates.y1 <= page_rect.height


def native_line_bbox(
    source: Path,
    page_number: int,
    text: str,
    *,
    occurrence: int = 0,
) -> PdfBounds:
    with pymupdf.open(source) as document:
        page = document[page_number - 1]
        matches = page.search_for(text)
        assert len(matches) > occurrence
        rect = matches[occurrence]
        return (rect.x0, rect.y0, rect.x1, rect.y1)


def pdf_bbox_to_ocr_polygon(
    source: Path, page_number: int, bbox: PdfBounds
) -> list[list[float]]:
    with pymupdf.open(source) as document:
        page = document[page_number - 1]
        pixmap = page.get_pixmap(dpi=300, colorspace=pymupdf.csRGB, alpha=False)
        scale_x = pixmap.width / page.rect.width
        scale_y = pixmap.height / page.rect.height
    x0, y0, x1, y1 = (
        bbox[0] * scale_x,
        bbox[1] * scale_y,
        bbox[2] * scale_x,
        bbox[3] * scale_y,
    )
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_fast_text_parser_preserves_native_text_and_coordinates(tmp_path: Path):
    source = make_pdf(tmp_path / "native.pdf", page_count=2)

    page = FastTextParser().parse_page(source, 2)

    assert page.status == "PARSED"
    assert page.page_number == 2
    text_blocks = [block for block in page.blocks if block.block_type == "TEXT"]
    assert len(text_blocks) == 1
    assert text_blocks[0].text == "First line\nSecond line"
    assert text_blocks[0].reading_order == 0
    assert text_blocks[0].source_coordinates.origin == "TOP_LEFT"
    assert_pdf_point_bounds(source, page)


def test_fast_text_parser_joins_spans_using_word_geometry(monkeypatch, tmp_path: Path):
    class FakePage:
        def get_contents(self) -> list[int]:
            return []

        def get_text(self, mode: str, *, sort: bool) -> dict[str, object]:
            assert mode == "dict"
            assert not sort
            return {
                "blocks": [
                    {
                        "type": 0,
                        "bbox": (10, 10, 100, 50),
                        "lines": [
                            {
                                "bbox": (10, 10, 60, 25),
                                "dir": (1.0, 0.0),
                                "spans": [
                                    {
                                        "text": "com",
                                        "bbox": (10, 10, 30, 25),
                                        "size": 12.0,
                                    },
                                    {
                                        "text": "pliance",
                                        "bbox": (30, 10, 60, 25),
                                        "size": 12.0,
                                    },
                                ],
                            },
                            {
                                "bbox": (10, 30, 100, 45),
                                "dir": (1.0, 0.0),
                                "spans": [
                                    {
                                        "text": "Boundary",
                                        "bbox": (10, 30, 50, 45),
                                        "size": 12.0,
                                    },
                                    {
                                        "text": "preserved",
                                        "bbox": (54, 30, 100, 45),
                                        "size": 12.0,
                                    },
                                ],
                            },
                        ],
                    }
                ]
            }

        def get_drawings(self) -> list[object]:
            return []

    class FakeDocument:
        needs_pass = False
        page_count = 1

        def __enter__(self):
            return self

        def __exit__(
            self, exc_type: object, exc_value: object, traceback: object
        ) -> bool:
            del exc_type, exc_value, traceback
            return False

        def load_page(self, page_index: int) -> FakePage:
            assert page_index == 0
            return FakePage()

    source = tmp_path / "mock.pdf"

    def fake_open(opened_source: Path) -> FakeDocument:
        assert opened_source == source
        return FakeDocument()

    monkeypatch.setattr("src.ingestion.parsers.pymupdf.open", fake_open)
    page = FastTextParser().parse_page(source, 1)

    text_blocks = [block for block in page.blocks if block.block_type == "TEXT"]
    assert len(text_blocks) == 1
    assert text_blocks[0].text == "compliance\nBoundary preserved"


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
    assert_pdf_point_bounds(vector_source, vector)


@pytest.mark.parametrize(
    ("coord_origin", "expected_origin"),
    [
        (CoordOrigin.TOPLEFT, "TOP_LEFT"),
        (CoordOrigin.BOTTOMLEFT, "BOTTOM_LEFT"),
    ],
)
def test_structured_parser_preserves_docling_coordinate_origin_and_bounds(
    tmp_path: Path,
    coord_origin: CoordOrigin,
    expected_origin: str,
):
    source = make_pdf(tmp_path / "structured.pdf", page_count=2)
    bbox = SimpleNamespace(
        l=10.0,
        t=90.0,
        r=300.0,
        b=70.0,
        coord_origin=coord_origin,
    )

    class Table:
        label = SimpleNamespace(value="table")
        text = None
        prov = [SimpleNamespace(page_no=2, bbox=bbox)]

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
            assert page_range == (2, 2)
            assert not raises_on_error
            return FakeResult()

    page = StructuredParser(converter=FakeConverter()).parse_page(source, 2)

    tables = [block for block in page.blocks if block.block_type == "TABLE"]
    assert page.page_number == 2
    assert tables
    assert "Contract review" in tables[0].text
    coordinates = tables[0].source_coordinates
    assert coordinates.origin == expected_origin
    assert coordinates.x0 == pytest.approx(10.0)
    assert coordinates.y0 == pytest.approx(70.0)
    assert coordinates.x1 == pytest.approx(300.0)
    assert coordinates.y1 == pytest.approx(90.0)
    assert_pdf_point_bounds(source, page)


def test_structured_parser_preserves_text_on_image_blocks(tmp_path: Path):
    source = make_pdf(tmp_path / "image-text.pdf")
    bbox = SimpleNamespace(
        l=10.0,
        t=90.0,
        r=300.0,
        b=70.0,
        coord_origin=CoordOrigin.TOPLEFT,
    )

    class Picture:
        label = SimpleNamespace(value="picture")
        text = "Text embedded in an image block"
        prov = [SimpleNamespace(page_no=1, bbox=bbox)]

    class FakeDocument:
        def iterate_items(self) -> Iterable[tuple[object, object]]:
            yield Picture(), 0

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

    assert len(page.blocks) == 1
    assert page.blocks[0].block_type == "IMAGE"
    assert page.blocks[0].text == "Text embedded in an image block"


def test_ocr_parser_uses_rapidocr_output_and_returns_canonical_blocks(tmp_path: Path):
    source = make_pdf(tmp_path / "ocr.pdf", page_count=2)
    native_bbox = native_line_bbox(source, 2, "First line")
    native_polygon = pdf_bbox_to_ocr_polygon(source, 2, native_bbox)

    class FakeRapidOCR:
        def __call__(self, image_path: str):
            assert Path(image_path).is_file()
            return SimpleNamespace(
                boxes=numpy.array([native_polygon]),
                txts=(" FIRST   LINE \nOCR-only line ",),
                scores=(0.95,),
            )

    page = OcrParser(
        engine="rapidocr",
        rapidocr_engine=FakeRapidOCR(),
    ).parse_page(source, 2)

    assert page.status == "PARSED"
    assert page.page_number == 2
    assert page.parser == "RAPIDOCR"
    all_text = [block.text for block in page.blocks if block.block_type == "TEXT"]
    assert sum("First line" in text for text in all_text) == 1
    assert sum("OCR-only line" in text for text in all_text) == 1
    ocr_block = next(block for block in page.blocks if "OCR-only line" in block.text)
    assert ocr_block.text == "OCR-only line"
    assert ocr_block.confidence == pytest.approx(95.0)
    assert_pdf_point_bounds(source, page)


def test_ocr_parser_deduplicates_repeated_ocr_lines_with_case_and_whitespace(
    tmp_path: Path,
):
    source = make_pdf(tmp_path / "ocr-duplicates.pdf")
    repeated_box = [[30, 40], [130, 40], [130, 60], [30, 60]]

    class FakeRapidOCR:
        def __call__(self, image_path: str):
            assert Path(image_path).is_file()
            return SimpleNamespace(
                boxes=numpy.array(
                    [
                        repeated_box,
                        repeated_box,
                    ]
                ),
                txts=("Unseen   OCR line", " unseen OCR\tline "),
                scores=(0.95, 0.9),
            )

    page = OcrParser(
        engine="rapidocr",
        rapidocr_engine=FakeRapidOCR(),
    ).parse_page(source, 1)

    recognized = [block.text for block in page.blocks if block.block_type == "TEXT"]
    assert (
        sum(
            " ".join(line.casefold().split()) == "unseen ocr line"
            for text in recognized
            for line in text.splitlines()
        )
        == 1
    )


def test_ocr_parser_deduplicates_by_text_and_location(tmp_path: Path):
    source = tmp_path / "repeated-native.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Repeated phrase")
    document.save(source)
    document.close()

    native_polygon = pdf_bbox_to_ocr_polygon(
        source, 1, native_line_bbox(source, 1, "Repeated phrase")
    )
    with pymupdf.open(source) as document:
        page_rect = document[0].rect
    remote_bbox = (
        page_rect.width * 0.5,
        page_rect.height * 0.5,
        page_rect.width * 0.75,
        page_rect.height * 0.5 + 24,
    )
    remote_polygon = pdf_bbox_to_ocr_polygon(source, 1, remote_bbox)

    class FakeRapidOCR:
        def __call__(self, image_path: str):
            assert Path(image_path).is_file()
            return SimpleNamespace(
                boxes=numpy.array([native_polygon, remote_polygon, remote_polygon]),
                txts=(
                    "Repeated phrase",
                    "Repeated phrase",
                    " REPEATED   PHRASE ",
                ),
                scores=(0.95, 0.9, 0.9),
            )

    page = OcrParser(
        engine="rapidocr",
        rapidocr_engine=FakeRapidOCR(),
    ).parse_page(source, 1)

    repeated_blocks = [
        block
        for block in page.blocks
        if block.block_type == "TEXT"
        and " ".join(block.text.casefold().split()) == "repeated phrase"
    ]
    assert len(repeated_blocks) == 2
    assert any(
        block.source_coordinates.x0 > page_rect.width * 0.45
        for block in repeated_blocks
    )


@pytest.mark.parametrize(
    ("horizontal_shift_fraction", "expected_text_blocks"),
    [
        (0.75, 2),
        (0.19, 1),
    ],
)
def test_ocr_parser_requires_significant_overlap_for_duplicate(
    tmp_path: Path,
    horizontal_shift_fraction: float,
    expected_text_blocks: int,
):
    source = tmp_path / "overlap.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Overlap phrase")
    document.save(source)
    document.close()

    native_bbox = native_line_bbox(source, 1, "Overlap phrase")
    width = native_bbox[2] - native_bbox[0]
    ocr_bbox = (
        native_bbox[0] + width * horizontal_shift_fraction,
        native_bbox[1],
        native_bbox[2] + width * horizontal_shift_fraction,
        native_bbox[3],
    )
    ocr_polygon = pdf_bbox_to_ocr_polygon(source, 1, ocr_bbox)

    class FakeRapidOCR:
        def __call__(self, image_path: str):
            assert Path(image_path).is_file()
            return SimpleNamespace(
                boxes=numpy.array([ocr_polygon]),
                txts=("Overlap phrase",),
                scores=(0.95,),
            )

    page_result = OcrParser(
        engine="rapidocr",
        rapidocr_engine=FakeRapidOCR(),
    ).parse_page(source, 1)

    text_blocks = [block for block in page_result.blocks if block.block_type == "TEXT"]
    assert len(text_blocks) == expected_text_blocks


def test_ocr_parser_deduplicates_matching_text_in_separate_table_cells(
    tmp_path: Path,
):
    source = tmp_path / "table-cells.pdf"
    document = pymupdf.open()
    page = document.new_page()
    cell_boxes = (
        pymupdf.Rect(60, 60, 250, 110),
        pymupdf.Rect(60, 160, 250, 210),
    )
    for cell_box in cell_boxes:
        page.draw_rect(cell_box)
        page.insert_text((cell_box.x0 + 8, cell_box.y0 + 25), "Yes")
    document.save(source)
    document.close()

    ocr_polygons = [
        pdf_bbox_to_ocr_polygon(
            source, 1, native_line_bbox(source, 1, "Yes", occurrence=index)
        )
        for index in range(2)
    ]

    class FakeRapidOCR:
        def __call__(self, image_path: str):
            assert Path(image_path).is_file()
            return SimpleNamespace(
                boxes=numpy.array(ocr_polygons),
                txts=("Yes", "Yes"),
                scores=(0.95, 0.95),
            )

    page_result = OcrParser(
        engine="rapidocr",
        rapidocr_engine=FakeRapidOCR(),
    ).parse_page(source, 1)
    native_yes = [
        line.strip()
        for block in page_result.blocks
        if block.block_type == "TEXT"
        for line in block.text.splitlines()
        if line.strip() == "Yes"
    ]

    assert len(native_yes) == 2


@pytest.mark.parametrize(
    ("span_texts", "span_bounds", "expected_text"),
    [
        (
            ("com", "pliance"),
            ((10, 10, 30, 25), (30, 10, 60, 25)),
            "compliance",
        ),
        (
            ("Boundary", "preserved"),
            ((10, 10, 50, 25), (54, 10, 100, 25)),
            "boundary preserved",
        ),
    ],
)
def test_native_text_geometry_preserves_word_boundaries_between_spans(
    span_texts: tuple[str, str],
    span_bounds: tuple[PdfBounds, PdfBounds],
    expected_text: str,
):
    class FakePage:
        def get_text(self, mode: str, *, sort: bool):
            assert mode == "dict"
            assert not sort
            return {
                "blocks": [
                    {
                        "type": 0,
                        "lines": [
                            {
                                "bbox": (10, 10, 100, 25),
                                "dir": (1.0, 0.0),
                                "spans": [
                                    {
                                        "text": text,
                                        "bbox": bbox,
                                        "size": 12.0,
                                    }
                                    for text, bbox in zip(
                                        span_texts, span_bounds, strict=True
                                    )
                                ],
                            }
                        ],
                    }
                ]
            }

    lines = _native_text_line_geometry(cast(pymupdf.Page, FakePage()))

    assert any(text == expected_text for text, _ in lines)


def test_ocr_parser_parses_tesseract_tsv(monkeypatch, tmp_path: Path):
    source = make_pdf(tmp_path / "tesseract.pdf", page_count=2)
    native_polygon = pdf_bbox_to_ocr_polygon(
        source, 2, native_line_bbox(source, 2, "First line")
    )
    left = int(native_polygon[0][0])
    top = int(native_polygon[0][1])
    width = max(1, int(native_polygon[2][0]) - left)
    height = max(1, int(native_polygon[2][1]) - top)

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
                f"5\t1\t1\t1\t2\t1\t{left}\t{top}\t{width}\t"
                f"{height}\t95.0\tFirst line\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("src.ingestion.parsers.subprocess.run", fake_run)
    page = OcrParser(engine="tesseract", tesseract_command="tesseract-test").parse_page(
        source, 2
    )

    ocr_blocks = [block for block in page.blocks if block.text == "OCR text"]
    text_blocks = [block for block in page.blocks if block.block_type == "TEXT"]
    assert page.page_number == 2
    assert len(ocr_blocks) == 1
    assert sum("First line" in block.text for block in text_blocks) == 1
    assert ocr_blocks[0].confidence == pytest.approx(91.5)
    assert ocr_blocks[0].source_coordinates.x0 == pytest.approx(2.4, abs=0.01)
    assert_pdf_point_bounds(source, page)


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
