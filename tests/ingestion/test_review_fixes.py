import asyncio
import base64
import json
import shutil

import httpx
import pymupdf
import pytest

from src.ingestion.parser_worker.limits import pdf_execution_timeout_seconds
from src.ingestion.scaledown_client import (
    ScaleDownClient,
    ScaleDownRateLimitError,
    ScaleDownResponseError,
    SCALEDOWN_CHAT_COMPLETIONS_PATH,
)


def test_timeout_scales_and_caps():
    assert [pdf_execution_timeout_seconds(n) for n in (1, 10, 500)] == [60, 600, 1800]


def test_blank_extract_text_is_returned_not_raised():
    request_payloads = []

    async def run():
        async def handle(request):
            request_payloads.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "r1",
                    "model": "extract-1",
                    "choices": [{"message": {"content": json.dumps({"ocr_text": " "})}}],
                },
            )

        transport = httpx.MockTransport(
            handle
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = ScaleDownClient(http_client=http, api_key="k")
            return await client.extract_document_text(b"%PDF-1.4")

    result = asyncio.run(run())
    assert result.ocr_text.strip() == ""
    request = request_payloads[0]
    payload = json.loads(request.content)
    schema = payload["response_format"][
        "json_schema"
    ]["schema"]
    assert request.url.path == SCALEDOWN_CHAT_COMPLETIONS_PATH
    assert payload["model"] == "extract-1"
    assert payload["document"] == base64.b64encode(b"%PDF-1.4").decode("ascii")
    assert payload["document_mime_type"] == "application/pdf"
    assert schema["properties"]["text_span"]["type"] == "string"
    assert "ocr_text" not in schema["properties"]


@pytest.mark.parametrize(
    ("content", "expected_text"),
    [
        ({"ocr_text": "Recognized text"}, "Recognized text"),
    ],
)
def test_extract_accepts_string_ocr_text_contract(content, expected_text):
    async def run():
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "model": "extract-1",
                    "choices": [
                        {"message": {"content": json.dumps(content)}}
                    ],
                },
            )
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = ScaleDownClient(http_client=http, api_key="k")
            return await client.extract_document_text(b"%PDF-1.4")

    assert asyncio.run(run()).ocr_text == expected_text


def test_extract_uses_documented_response_and_preserves_raw_ocr_text():
    raw_ocr_text = "Full first paragraph.\nFull second paragraph."
    captured_requests = []

    async def run():
        async def handle(request):
            captured_requests.append(request)
            return httpx.Response(
                200,
                headers={"x-request-id": "document-extract-123"},
                json={
                    "model": "extract-1",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "entities": [
                                            {
                                                "text": "first paragraph",
                                                "type": "text_span",
                                                "confidence": 0.97,
                                                "start": 5,
                                                "end": 20,
                                                "context": "an entity context",
                                            }
                                        ],
                                        "structured_result": None,
                                        "input_tokens": 142,
                                        "ocr_text": raw_ocr_text,
                                    }
                                )
                            }
                        }
                    ],
                },
            )

        transport = httpx.MockTransport(handle)
        async with httpx.AsyncClient(transport=transport) as http:
            client = ScaleDownClient(http_client=http, api_key="mock-key")
            return await client.extract_document_text(
                b"%PDF-1.4", document_mime_type="application/pdf"
            )

    result = asyncio.run(run())
    request = captured_requests[0]
    request_body = json.loads(request.content)

    assert request.url.path == SCALEDOWN_CHAT_COMPLETIONS_PATH
    assert request_body["model"] == "extract-1"
    assert request_body["document"] == base64.b64encode(b"%PDF-1.4").decode("ascii")
    assert request_body["document_mime_type"] == "application/pdf"
    assert result.ocr_text == raw_ocr_text
    assert result.ocr_text != "first paragraph an entity context"
    assert result.request_id == "document-extract-123"


def test_extract_maps_empty_provider_result_to_no_text():
    async def run():
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "model": "extract-1",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "entities": [],
                                        "input_tokens": 10,
                                        "ocr_text": None,
                                        "structured_result": None,
                                    }
                                )
                            }
                        }
                    ],
                },
            )
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = ScaleDownClient(http_client=http, api_key="k")
            return await client.extract_document_text(b"%PDF-1.4")

    assert asyncio.run(run()).ocr_text == ""


def test_extract_rejects_non_string_ocr_text_without_coercion():
    async def run():
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "model": "extract-1",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({"ocr_text": ["Text"]})
                            }
                        }
                    ],
                },
            )
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = ScaleDownClient(http_client=http, api_key="k")
            return await client.extract_document_text(b"%PDF-1.4")

    with pytest.raises(
        ScaleDownResponseError, match="received list"
    ):
        asyncio.run(run())


def test_extract_error_reports_response_shape_without_ocr_values():
    async def run():
        response_content = {
            "entities": [
                {
                    "confidence": 0.97,
                    "context": "Case report text surrounding the finding.",
                    "end": 18,
                    "start": 0,
                    "text": "Case report text",
                    "type": "text_span",
                }
            ],
            "ocr_text": None,
            "structured_result": None,
            "input_tokens": 142,
        }
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"x-request-id": "extract-request-123"},
                json={
                    "model": "extract-1",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(response_content)
                            }
                        }
                    ],
                },
            )
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = ScaleDownClient(http_client=http, api_key="k")
            return await client.extract_document_text(b"%PDF-1.4")

    with pytest.raises(ScaleDownResponseError) as error:
        asyncio.run(run())
    assert "received NoneType" in str(error.value)
    assert "ocr_text: NoneType" in str(error.value)
    assert "entities[0]: confidence: float" in str(error.value)
    assert error.value.request_id == "extract-request-123"
    assert error.value.http_status == 200


def test_extract_http_failure_preserves_status_and_provider_request_id():
    async def run():
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                429,
                headers={"x-amzn-requestid": "rate-limit-request-789"},
                json={"error": "rate limited"},
            )
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = ScaleDownClient(http_client=http, api_key="k")
            return await client.extract_document_text(b"%PDF-1.4")

    with pytest.raises(ScaleDownRateLimitError) as error:
        asyncio.run(run())
    assert error.value.http_status == 429
    assert error.value.request_id == "rate-limit-request-789"


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
