import asyncio
from pathlib import Path

from src.ingestion.contracts import DocumentRoutePlan, PageRouteDecision
from src.ingestion.chunking import prepare_document_for_chunking
from src.ingestion.parser_contracts import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalPage,
    ParserFailure,
    SourceCoordinates,
)
from src.ingestion.scaledown_client import ScaleDownExtractResult
from src.ingestion.scaledown_extraction import recover_escalated_pages


class PageExporter:
    async def extract_page_pdf(
        self, source: Path, page_number: int
    ) -> tuple[bytes, float, float]:
        return b"page-pdf", 612.0, 792.0


class TextExtractor:
    def __init__(self, text: str):
        self.text = text

    async def extract_document_text(
        self,
        document: bytes,
        *,
        document_mime_type: str = "application/pdf",
    ) -> ScaleDownExtractResult:
        assert document == b"page-pdf"
        assert document_mime_type == "application/pdf"
        return ScaleDownExtractResult(
            ocr_text=self.text,
            model_id="extract-1",
            request_id="request-1",
        )


def _inputs():
    failure = ParserFailure(
        parser="DISPATCHER",
        code="ROUTE_NOT_PARSABLE",
        message="Short native text requires fallback extraction.",
    )
    canonical = CanonicalDocument(
        page_count=1,
        document_disposition="PROCESS",
        pages=(
            CanonicalPage(
                page_number=1,
                parser="DISPATCHER",
                status="FAILED",
                failure=failure,
            ),
        ),
    )
    plan = DocumentRoutePlan(
        page_count=1,
        decisions=(
            PageRouteDecision(
                page_number=1,
                route="ESCALATE",
                reason="Short native text requires extraction fallback.",
                decision_source="RULES",
                policy_version="router-test-1",
            ),
        ),
        document_disposition="PROCESS",
    )
    return canonical, plan


def test_escalated_short_text_page_is_recovered_with_source_failure_visible():
    canonical, plan = _inputs()

    recovered = asyncio.run(
        recover_escalated_pages(
            source=None,
            canonical=canonical,
            route_plan=plan,
            page_exporter=PageExporter(),
            text_extractor=TextExtractor("Dated: 5 March 2026\nPlace: Delhi"),
        )
    )

    page = recovered.pages[0]
    assert page.status == "PARSED"
    assert page.parser == "SCALEDOWN_EXTRACT"
    assert page.blocks[0].text == "Dated: 5 March 2026\nPlace: Delhi"
    assert page.blocks[0].source_coordinates.x1 == 612.0
    assert page.warnings[0].code == "ROUTE_NOT_PARSABLE"
    assert page.warnings[1].code == "FALLBACK_TEXT_RECOVERED"


def test_escalated_page_with_no_extracted_text_is_marked_blank():
    canonical, plan = _inputs()

    recovered = asyncio.run(
        recover_escalated_pages(
            source=None,
            canonical=canonical,
            route_plan=plan,
            page_exporter=PageExporter(),
            text_extractor=TextExtractor("  \n"),
        )
    )

    page = recovered.pages[0]
    assert page.status == "BLANK"
    assert page.blocks == ()
    assert page.warnings[-1].code == "FALLBACK_NO_TEXT"


def test_no_text_fallback_preserves_image_blocks_for_crop_without_chunking():
    _, plan = _inputs()
    canonical = CanonicalDocument(
        page_count=1,
        document_disposition="PROCESS",
        pages=(
            CanonicalPage(
                page_number=1,
                parser="TESSERACT",
                status="PARSED",
                blocks=(
                    CanonicalBlock(
                        page_number=1,
                        text="",
                        block_type="IMAGE",
                        reading_order=0,
                        source_coordinates=SourceCoordinates(
                            x0=0,
                            y0=0,
                            x1=612,
                            y1=792,
                            unit="PDF_POINT",
                            origin="TOP_LEFT",
                        ),
                    ),
                ),
            ),
        ),
    )

    recovered = asyncio.run(
        recover_escalated_pages(
            source=Path("unused.pdf"),
            canonical=canonical,
            route_plan=plan,
            page_exporter=PageExporter(),
            text_extractor=TextExtractor(""),
        )
    )

    page = recovered.pages[0]
    assert page.status == "PARSED"
    assert len(page.blocks) == 1
    assert page.blocks[0].block_type == "IMAGE"
    assert page.warnings[-1].code == "FALLBACK_NO_TEXT"
    assert (
        prepare_document_for_chunking(recovered).pages[0].outcome
        == "SKIPPED_IMAGE_NO_TEXT"
    )


def test_escalated_page_with_native_text_does_not_call_extract():
    _, plan = _inputs()
    canonical = CanonicalDocument(
        page_count=1,
        document_disposition="PROCESS",
        pages=(
            CanonicalPage(
                page_number=1,
                parser="FAST_TEXT",
                status="PARSED",
                blocks=(
                    {
                        "page_number": 1,
                        "text": "Dated: 5 March 2026",
                        "block_type": "TEXT",
                        "reading_order": 0,
                        "source_coordinates": {
                            "x0": 0,
                            "y0": 0,
                            "x1": 612,
                            "y1": 792,
                            "unit": "PDF_POINT",
                            "origin": "TOP_LEFT",
                        },
                    },
                ),
            ),
        ),
    )

    class UnexpectedExtractor:
        async def extract_document_text(self, document, *, document_mime_type):
            raise AssertionError("Native text should avoid the external OCR call")

    recovered = asyncio.run(
        recover_escalated_pages(
            source=None,
            canonical=canonical,
            route_plan=plan,
            page_exporter=PageExporter(),
            text_extractor=UnexpectedExtractor(),
        )
    )
    assert recovered.pages[0].blocks[0].text == "Dated: 5 March 2026"


def test_ocr_page_with_empty_image_block_is_recovered():
    _, escalate_plan = _inputs()
    plan = DocumentRoutePlan(
        page_count=1,
        decisions=(
            PageRouteDecision(
                page_number=1,
                route="OCR",
                reason="Scanned page",
                decision_source="RULES",
                policy_version="router-test-1",
            ),
        ),
        document_disposition="PROCESS",
    )
    canonical = CanonicalDocument(
        page_count=1,
        document_disposition="PROCESS",
        pages=(
            CanonicalPage(
                page_number=1,
                parser="TESSERACT",
                status="PARSED",
                blocks=(
                    {
                        "page_number": 1,
                        "text": "",
                        "block_type": "IMAGE",
                        "reading_order": 0,
                        "source_coordinates": {
                            "x0": 0,
                            "y0": 0,
                            "x1": 612,
                            "y1": 792,
                            "unit": "PDF_POINT",
                            "origin": "TOP_LEFT",
                        },
                    },
                ),
            ),
        ),
    )
    assert escalate_plan.decisions[0].route == "ESCALATE"

    recovered = asyncio.run(
        recover_escalated_pages(
            source=None,
            canonical=canonical,
            route_plan=plan,
            page_exporter=PageExporter(),
            text_extractor=TextExtractor("OCR recovered"),
        )
    )
    assert recovered.pages[0].blocks[0].text == "OCR recovered"
