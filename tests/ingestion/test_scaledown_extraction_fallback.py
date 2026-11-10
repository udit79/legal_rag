import asyncio

import pytest

from src.ingestion.contracts import DocumentRoutePlan, PageRouteDecision
from src.ingestion.parser_contracts import (
    CanonicalDocument,
    CanonicalPage,
    ParserFailure,
)
from src.ingestion.scaledown_client import ScaleDownExtractResult, ScaleDownResponseError
from src.ingestion.scaledown_extraction import recover_escalated_pages


class PageExporter:
    async def extract_page_pdf(self, source, page_number):
        return b"page-pdf", 612.0, 792.0


class TextExtractor:
    def __init__(self, text: str):
        self.text = text

    async def extract_document_text(self, document, *, document_mime_type):
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


def test_escalated_page_with_no_extracted_text_fails_explicitly():
    canonical, plan = _inputs()

    with pytest.raises(ScaleDownResponseError, match="blank text"):
        asyncio.run(
            recover_escalated_pages(
                source=None,
                canonical=canonical,
                route_plan=plan,
                page_exporter=PageExporter(),
                text_extractor=TextExtractor("  \n"),
            )
        )
