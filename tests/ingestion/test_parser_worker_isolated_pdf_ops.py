import base64

import pymupdf

from src.ingestion.parser_contracts import SourceCoordinates
from src.ingestion.parser_worker import worker
from src.ingestion.parser_worker.result import (
    ImageCropRequest,
    ImageCropSpec,
    PdfInspectionRequest,
)
from src.ingestion.parser_worker.limits import pdf_execution_timeout_seconds


def _pdf(path):
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_rect(pymupdf.Rect(10, 10, 100, 80), color=(1, 0, 0), fill=(1, 0, 0))
    document.save(path)
    document.close()


def test_pdf_execution_timeout_scales_with_document_size_and_is_capped():
    assert pdf_execution_timeout_seconds(1) == 60
    assert pdf_execution_timeout_seconds(10) == 600
    assert pdf_execution_timeout_seconds(500) == 1800


def test_worker_pdf_inspection_returns_page_and_object_counts(tmp_path, monkeypatch):
    path = tmp_path / "source.pdf"
    _pdf(path)
    monkeypatch.setattr(worker, "_pdf_path", lambda _filename: path)
    request = PdfInspectionRequest(
        request_id="a" * 32,
        filename=f"{'a' * 32}.pdf",
        max_pdf_objects=200_000,
    )

    result = worker._handle_pdf_inspection(request.model_dump_json().encode())

    assert result.inspection_status == "OK"
    assert result.page_count == 1
    assert result.object_count > 0


def test_worker_crops_an_image_region_and_returns_provenance(tmp_path, monkeypatch):
    path = tmp_path / "source.pdf"
    _pdf(path)
    monkeypatch.setattr(worker, "_pdf_path", lambda _filename: path)
    request = ImageCropRequest(
        request_id="b" * 32,
        filename=f"{'b' * 32}.pdf",
        images=(
            ImageCropSpec(
                page_number=1,
                block_id="page-1-block-0",
                reading_order=0,
                source_coordinates=SourceCoordinates(
                    x0=10,
                    y0=10,
                    x1=100,
                    y1=80,
                    unit="PDF_POINT",
                    origin="TOP_LEFT",
                ),
            ),
        ),
    )

    result = worker._handle_crop_images(request.model_dump_json().encode())

    image = result.images[0]
    crop_bytes = base64.b64decode(image.data_base64)
    assert crop_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert image.block_id == "page-1-block-0"
    assert image.source_coordinates == request.images[0].source_coordinates
