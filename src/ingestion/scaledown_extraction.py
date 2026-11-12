from __future__ import annotations
from pathlib import Path
from typing import Protocol
from src.ingestion.contracts import DocumentRoutePlan
from src.ingestion.parser_contracts import CanonicalBlock,CanonicalDocument,CanonicalPage,ParserFailure,SourceCoordinates
from src.ingestion.scaledown_client import ScaleDownExtractResult

class _PagePdfExporter(Protocol):
    async def extract_page_pdf(
        self, source: Path, page_number: int
    ) -> tuple[bytes, float, float]: ...

class _PageTextExtractor(Protocol):
    async def extract_document_text(
        self, document: bytes, *, document_mime_type: str = "application/pdf"
    ) -> ScaleDownExtractResult: ...

async def recover_escalated_pages(
    source: Path,
    canonical: CanonicalDocument,
    route_plan: DocumentRoutePlan,
    *,
    page_exporter: _PagePdfExporter,
    text_extractor: _PageTextExtractor,
) -> CanonicalDocument:
    """Recover ESCALATE pages and failed local OCR pages using Extract.

    The PDF page is isolated by the parser worker before it is sent to ScaleDown.
    Extract's raw OCR output is kept as a single canonical text block with
    page-level geometry; it does not provide word-level bounding boxes.
    """
    if canonical.page_count != route_plan.page_count:
        raise ValueError("Canonical document and route plan page counts differ")

    recovered_pages: list[CanonicalPage] = []
    for page, decision in zip(canonical.pages, route_plan.decisions, strict=True):
        has_text = any(
            block.block_type in {"TEXT", "TABLE"} and block.text.strip()
            for block in page.blocks
        )
        should_recover = decision.route in {"ESCALATE", "OCR", "LAYOUT_OCR"} and (
            page.status in {"FAILED", "BLANK", "VECTOR_ONLY"} or not has_text
        )
        if not should_recover:
            recovered_pages.append(page)
            continue

        page_pdf, width, height = await page_exporter.extract_page_pdf(
            source, page.page_number
        )
        extracted = await text_extractor.extract_document_text(
            page_pdf,
            document_mime_type="application/pdf",
        )
        prior_failure = page.failure or ParserFailure(
            parser=decision.route,
            code="LOCAL_OCR_NO_TEXT",
            message="The local route produced no usable text.",
        )
        if not extracted.ocr_text.strip():
            no_text_warning = ParserFailure(
                parser="SCALEDOWN_EXTRACT",
                code="FALLBACK_NO_TEXT",
                message=(
                    f"ScaleDown Extract ({extracted.model_id}) found no text; "
                    "the page was excluded from text chunking."
                ),
            )
            if any(
                block.block_type == "IMAGE" and not block.text.strip()
                for block in page.blocks
            ):
                recovered_pages.append(
                    page.model_copy(
                        update={
                            "warnings": page.warnings
                            + (prior_failure, no_text_warning)
                        }
                    )
                )
                continue
            recovered_pages.append(
                CanonicalPage(
                    page_number=page.page_number,
                    parser="SCALEDOWN_EXTRACT",
                    status="BLANK",
                    warnings=page.warnings + (prior_failure, no_text_warning),
                )
            )
            continue

        fallback_warning = ParserFailure(
            parser="SCALEDOWN_EXTRACT",
            code="FALLBACK_TEXT_RECOVERED",
            message=(
                f"Page text was recovered with {extracted.model_id}"
                f" (request {extracted.request_id or 'unavailable'}); geometry is "
                "page-level and does not identify individual word positions."
            ),
        )
        recovered_pages.append(
            CanonicalPage(
                page_number=page.page_number,
                parser="SCALEDOWN_EXTRACT",
                status="PARSED",
                blocks=(
                    CanonicalBlock(
                        page_number=page.page_number,
                        text=extracted.ocr_text,
                        block_type="TEXT",
                        reading_order=0,
                        source_coordinates=SourceCoordinates(
                            x0=0.0,
                            y0=0.0,
                            x1=width,
                            y1=height,
                            unit="PDF_POINT",
                            origin="TOP_LEFT",
                        ),
                    ),
                ),
                warnings=page.warnings + (prior_failure, fallback_warning),
            )
        )

    return canonical.model_copy(update={"pages": tuple(recovered_pages)})
