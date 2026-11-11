import asyncio
import json
import shutil

import httpx
import pymupdf
import pytest

from src.ingestion.parser_worker.limits import pdf_execution_timeout_seconds
from src.ingestion.scaledown_client import ScaleDownClient


def test_timeout_scales_and_caps():
    assert [pdf_execution_timeout_seconds(n) for n in (1, 10, 500)] == [60, 600, 1800]


def test_blank_extract_text_is_returned_not_raised():
    async def run():
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "id": "r1",
                    "model": "extract-1",
                    "choices": [{"message": {"content": json.dumps({"ocr_text": " "})}}],
                },
            )
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = ScaleDownClient(http_client=http, api_key="k")
            return await client.extract_document_text(b"%PDF-1.4")

    assert asyncio.run(run()).ocr_text.strip() == ""


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_two_column_ocr_keeps_column_reading_order(tmp_path):
    from src.ingestion.dispatcher import dispatch_document
    from src.ingestion.preflight import inspect_pdf
    from src.ingestion.router import route_document

    source = pymupdf.open()
    page = source.new_page()
    page.insert_textbox(pymupdf.Rect(36, 50, 290, 800), "\n".join(
        f"LEFT col line {i}: the appellant relied upon the earlier decision." for i in range(1, 26)), fontsize=9)
    page.insert_textbox(pymupdf.Rect(310, 50, 560, 800), "\n".join(
        f"RIGHT col line {i}: the respondent submitted that notice was served." for i in range(1, 26)), fontsize=9)
    scan = pymupdf.open()
    target = scan.new_page()
    target.insert_image(target.rect, pixmap=page.get_pixmap(dpi=200))
    path = tmp_path / "scan.pdf"
    scan.save(path)

    features = inspect_pdf(path)
    canonical = dispatch_document(path, route_document(features, "v"))
    sides = [
        "L" if b.text.startswith("LEFT") else "R"
        for b in canonical.pages[0].blocks
        if b.block_type == "TEXT" and b.text.startswith(("LEFT", "RIGHT"))
    ]
    assert "L" in sides and "R" in sides
    assert "".join(sides).rstrip("R").count("R") == 0  # no R before the last L
