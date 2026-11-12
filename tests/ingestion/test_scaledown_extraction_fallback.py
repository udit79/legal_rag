import asyncio
import logging
from pathlib import Path

import pytest

from src.ingestion.contracts import DocumentRoutePlan, PageRouteDecision
from src.ingestion.chunking import prepare_document_for_chunking
from src.ingestion.parser_contracts import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalPage,
    ParserFailure,
    SourceCoordinates,
)
from src.ingestion.scaledown_client import (
    ScaleDownExtractResult,
    ScaleDownResponseError,
)
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
    assert page.status == "PARSED"
    assert len(page.blocks) == 1
    assert page.blocks[0].block_type == "IMAGE"
    assert page.blocks[0].source_coordinates.x1 == 612
    assert page.blocks[0].source_coordinates.y1 == 792
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


def test_page_recovery_runs_concurrently_with_a_bound_and_keeps_page_order():
    page_count = 4
    canonical = CanonicalDocument(
        page_count=page_count,
        document_disposition="PROCESS",
        pages=tuple(
            CanonicalPage(
                page_number=page_number,
                parser="DISPATCHER",
                status="BLANK",
            )
            for page_number in range(1, page_count + 1)
        ),
    )
    plan = DocumentRoutePlan(
        page_count=page_count,
        decisions=tuple(
            PageRouteDecision(
                page_number=page_number,
                route="ESCALATE",
                reason="Test async page recovery",
                decision_source="RULES",
                policy_version="router-test-1",
            )
            for page_number in range(1, page_count + 1)
        ),
        document_disposition="PROCESS",
    )

    class NumberedPageExporter:
        async def extract_page_pdf(self, source, page_number):
            return f"page-{page_number}".encode(), 612.0, 792.0

    class ConcurrentTextExtractor:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def extract_document_text(
            self, document, *, document_mime_type="application/pdf"
        ):
            page_number = int(document.decode().split("-")[1])
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep((page_count - page_number + 1) * 0.005)
            self.active -= 1
            return ScaleDownExtractResult(
                ocr_text=f"text-{page_number}",
                model_id="extract-1",
                request_id=f"request-{page_number}",
            )

    extractor = ConcurrentTextExtractor()
    recovered = asyncio.run(
        recover_escalated_pages(
            source=Path("unused.pdf"),
            canonical=canonical,
            route_plan=plan,
            page_exporter=NumberedPageExporter(),
            text_extractor=extractor,
        )
    )

    assert extractor.max_active == 2
    assert [page.page_number for page in recovered.pages] == [1, 2, 3, 4]
    assert [page.blocks[0].text for page in recovered.pages] == [
        "text-1",
        "text-2",
        "text-3",
        "text-4",
    ]


def test_scaledown_page_failure_logs_only_safe_diagnostic_fields(caplog):
    canonical, plan = _inputs()

    class SensitivePageExporter:
        async def extract_page_pdf(self, source, page_number):
            return b"secret-pdf-content", 612.0, 792.0

    class FailedTextExtractor:
        async def extract_document_text(
            self, document, *, document_mime_type="application/pdf"
        ):
            raise ScaleDownResponseError(
                "ScaleDown Extract did not return raw OCR text as a string; "
                "received NoneType",
                http_status=200,
                request_id="extract-request-456",
            )

    with caplog.at_level(
        logging.ERROR, logger="src.ingestion.scaledown_extraction"
    ):
        with pytest.raises(ScaleDownResponseError):
            asyncio.run(
                recover_escalated_pages(
                    source=Path("security_report.pdf"),
                    canonical=canonical,
                    route_plan=plan,
                    page_exporter=SensitivePageExporter(),
                    text_extractor=FailedTextExtractor(),
                )
            )

    record = next(
        record
        for record in caplog.records
        if record.message == "ScaleDown page recovery failed"
    )
    assert record.scaledown_exception_type == "ScaleDownResponseError"
    assert record.scaledown_http_status == 200
    assert record.scaledown_request_id == "extract-request-456"
    assert record.scaledown_page_number == 1
    assert "did not return raw OCR text" in record.scaledown_error_message
    assert "secret-pdf-content" not in caplog.text
