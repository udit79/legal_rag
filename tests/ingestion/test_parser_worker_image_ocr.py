import base64
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from src.ingestion.parser_worker.result import ImageOcrRequest
from src.ingestion.parser_worker.worker import _handle_image_ocr


def test_worker_ocr_protocol_checks_png_and_returns_requested_image_ids():
    buffer = BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="PNG")
    png = base64.b64encode(buffer.getvalue()).decode("ascii")
    request = ImageOcrRequest(
        request_id="a" * 32,
        images=({"image_id": "1:page-1-block-0", "data_base64": png},),
    )
    fake_ocr = SimpleNamespace(
        _tesseract=lambda _path: [("Scanned words", (0, 0, 1, 1), 99.0)]
    )

    response = _handle_image_ocr(
        request.model_dump_json().encode(), (None, None, None, fake_ocr)
    )

    assert response.type == "ocr_result"
    assert response.request_id == request.request_id
    assert response.results[0].image_id == "1:page-1-block-0"
    assert response.results[0].text == "Scanned words"


def test_worker_rejects_invalid_image_payload():
    request = ImageOcrRequest(
        request_id="b" * 32,
        images=({"image_id": "1:page-1-block-0", "data_base64": "not base64"},),
    )

    response = _handle_image_ocr(
        request.model_dump_json().encode(), (None, None, None, SimpleNamespace())
    )

    assert response.type == "error"
    assert response.error_code == "IMAGE_OCR_FAILED"
