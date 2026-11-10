"""Long-lived JSON-lines parser service; stdout is reserved for framed JSON."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

import pymupdf
from pydantic import BaseModel

from src.ingestion.dispatcher import dispatch_document
from src.ingestion.parsers import FastTextParser, OcrParser, StructuredParser
from src.ingestion.preflight import inspect_pdf
from src.ingestion.router import route_document
from src.ingestion.parser_worker.limits import (
    MAX_PDF_PAGES,
    MAX_PDF_SIZE_BYTES,
    MAX_WORKER_RESPONSE_BYTES,
    MAX_OCR_BATCH_BYTES,
    MAX_OCR_BATCH_IMAGES,
    MAX_OCR_IMAGE_BYTES,
    MAX_SINGLE_PAGE_PDF_BYTES,
    WORKER_MOUNT_PATH,
)
from src.ingestion.parser_worker.result import (
    ImageOcrRequest,
    ImageCropRequest,
    ImageCropResponse,
    ImageOcrResponse,
    ImageOcrText,
    ParseWorkerRequest,
    ParseWorkerResponse,
    PagePdfRequest,
    PagePdfResponse,
    PdfInspectionRequest,
    PdfInspectionResponse,
)


def _read_exact(stream: BinaryIO, size: int) -> bytes | None:
    parts = bytearray()
    while len(parts) < size:
        data = stream.read(size - len(parts))
        if not data:
            if not parts:
                return None
            raise EOFError("Incomplete parser-worker frame")
        parts.extend(data)
    return bytes(parts)


def _read_message(stream: BinaryIO) -> bytes | None:
    header = _read_exact(stream, 4)
    if header is None:
        return None
    size = int.from_bytes(header, "big")
    if size <= 0 or size > MAX_WORKER_RESPONSE_BYTES:
        raise ValueError("Parser-worker request frame has an invalid size")
    payload = _read_exact(stream, size)
    if payload is None:
        raise EOFError("Parser-worker request frame is incomplete")
    return payload


def _write_message(stream: BinaryIO, response: BaseModel) -> None:
    payload = response.model_dump_json().encode("utf-8")
    if len(payload) > MAX_WORKER_RESPONSE_BYTES:
        response = ParseWorkerResponse(
            type="error",
            request_id=response.request_id,
            error_code="RESULT_TOO_LARGE",
            error_type="ParserResultLimitExceeded",
        )
        payload = response.model_dump_json().encode("utf-8")
    stream.write(len(payload).to_bytes(4, "big"))
    stream.write(payload)
    stream.flush()


def _pdf_path(filename: str) -> Path:
    root = Path(WORKER_MOUNT_PATH).resolve()
    path = (root / filename).resolve()
    if path.parent != root or path.suffix.casefold() != ".pdf":
        raise ValueError("Worker input must be a PDF inside the mounted work folder")
    size = path.stat().st_size
    if size <= 0 or size > MAX_PDF_SIZE_BYTES:
        raise ValueError("Worker PDF size is outside configured limits")
    with path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            raise ValueError("Worker input does not have PDF magic bytes")
    return path


def _initialize_parsers() -> tuple[FastTextParser, StructuredParser, StructuredParser, OcrParser]:
    fast_text = FastTextParser()
    structured = StructuredParser()
    layout_ocr = StructuredParser(do_ocr=True)
    ocr = OcrParser()
    # Keep startup messages off the framed protocol stream.
    with redirect_stdout(sys.stderr):
        structured.initialize()
        layout_ocr.initialize()
    return fast_text, structured, layout_ocr, ocr


def _handle(
    raw_request: bytes,
    parsers: tuple[FastTextParser, StructuredParser, StructuredParser, OcrParser],
) -> ParseWorkerResponse | ImageOcrResponse | PagePdfResponse | PdfInspectionResponse | ImageCropResponse:
    request_id: str | None = None
    try:
        message_type = json.loads(raw_request).get("type")
        if message_type == "ocr_images":
            return _handle_image_ocr(raw_request, parsers)
        if message_type == "page_pdf":
            return _handle_page_pdf(raw_request)
        if message_type == "inspect_pdf":
            return _handle_pdf_inspection(raw_request)
        if message_type == "crop_images":
            return _handle_crop_images(raw_request)
        request = ParseWorkerRequest.model_validate_json(raw_request)
        request_id = request.request_id
        source = _pdf_path(request.filename)
        features = inspect_pdf(source)
        if features.page_count > MAX_PDF_PAGES:
            raise ValueError("Worker PDF page count is outside configured limits")
        route_plan = route_document(features, request.routing_policy_version)
        with redirect_stdout(sys.stderr):
            canonical = dispatch_document(
                source,
                route_plan,
                fast_text_parser=parsers[0],
                structured_parser=parsers[1],
                layout_ocr_parser=parsers[2],
                ocr_parser=parsers[3],
            )
        return ParseWorkerResponse(
            type="result",
            request_id=request_id,
            features=features,
            route_plan=route_plan,
            canonical=canonical,
        )
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return ParseWorkerResponse(
            type="error",
            request_id=request_id or "0" * 32,
            error_code="PARSER_WORKER_FAILED",
            error_type=type(exc).__name__,
        )


def _handle_pdf_inspection(raw_request: bytes) -> PdfInspectionResponse:
    request_id = "0" * 32
    try:
        request = PdfInspectionRequest.model_validate_json(raw_request)
        request_id = request.request_id
        source = _pdf_path(request.filename)
        with pymupdf.open(source) as document:
            page_count = document.page_count
            object_count = document.xref_length()
            active_content_xrefs = []
            if (
                page_count <= MAX_PDF_PAGES
                and object_count <= request.max_pdf_objects
            ):
                active_token = re.compile(
                    r"/(?:JavaScript|JS|OpenAction|AA|Launch|EmbeddedFiles)\b"
                )
                for xref in range(1, object_count):
                    if active_token.search(document.xref_object(xref, compressed=False)):
                        active_content_xrefs.append(xref)
                if document.embfile_count() > 0:
                    active_content_xrefs.append(-1)
            return PdfInspectionResponse(
                type="inspect_result",
                request_id=request_id,
                inspection_status="OK",
                page_count=page_count,
                object_count=object_count,
                active_content_xrefs=tuple(active_content_xrefs),
                encrypted=document.needs_pass,
            )
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return PdfInspectionResponse(
            type="inspect_result",
            request_id=request_id,
            inspection_status="FAILED",
            error_code=type(exc).__name__,
        )


def _handle_crop_images(raw_request: bytes) -> ImageCropResponse | ParseWorkerResponse:
    request_id = "0" * 32
    try:
        request = ImageCropRequest.model_validate_json(raw_request)
        request_id = request.request_id
        if len(request.images) > MAX_OCR_BATCH_IMAGES:
            raise ValueError("Image crop count exceeds the worker limit")
        identities = [(item.page_number, item.block_id) for item in request.images]
        if len(set(identities)) != len(identities):
            raise ValueError("Image crop identities must be unique")
        source = _pdf_path(request.filename)
        outputs = []
        total_encoded_bytes = 0
        with pymupdf.open(source) as pdf:
            for item in request.images:
                if item.page_number > pdf.page_count:
                    raise ValueError("Image crop page is outside the PDF")
                coordinates = item.source_coordinates
                if coordinates.unit != "PDF_POINT":
                    raise ValueError("Image crop coordinates must use PDF points")
                page = pdf.load_page(item.page_number - 1)
                if coordinates.origin == "BOTTOM_LEFT":
                    top = page.rect.height - coordinates.y1
                    bottom = page.rect.height - coordinates.y0
                else:
                    top, bottom = coordinates.y0, coordinates.y1
                clip = pymupdf.Rect(
                    coordinates.x0, top, coordinates.x1, bottom
                ) & page.rect
                if clip.is_empty or clip.width <= 0 or clip.height <= 0:
                    raise ValueError("Image block has no visible page area")
                scale = min(2.0, math.sqrt(4_000_000 / (clip.width * clip.height)))
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False
                )
                encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
                total_encoded_bytes += len(encoded)
                if total_encoded_bytes > MAX_OCR_BATCH_BYTES:
                    raise ValueError("Image crop response exceeds the worker limit")
                outputs.append(
                    {
                        **item.model_dump(),
                        "filename": f"page-{item.page_number}-{item.block_id}.png",
                        "data_base64": encoded,
                    }
                )
        return ImageCropResponse(
            type="crop_result",
            request_id=request_id,
            images=tuple(outputs),
        )
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return ParseWorkerResponse(
            type="error",
            request_id=request_id,
            error_code="IMAGE_CROP_FAILED",
            error_type=type(exc).__name__,
        )
def _handle_page_pdf(raw_request: bytes) -> PagePdfResponse | ParseWorkerResponse:
    request_id = "0" * 32
    try:
        request = PagePdfRequest.model_validate_json(raw_request)
        request_id = request.request_id
        source = _pdf_path(request.filename)
        with pymupdf.open(source) as document:
            if request.page_number > document.page_count:
                raise ValueError("Requested page is outside the PDF")
            page = document.load_page(request.page_number - 1)
            width_points = float(page.rect.width)
            height_points = float(page.rect.height)
            single_page = pymupdf.open()
            try:
                single_page.insert_pdf(
                    document,
                    from_page=request.page_number - 1,
                    to_page=request.page_number - 1,
                )
                pdf_bytes = single_page.tobytes(garbage=3, deflate=True)
            finally:
                single_page.close()
        if not pdf_bytes or len(pdf_bytes) > MAX_SINGLE_PAGE_PDF_BYTES:
            raise ValueError("Single-page PDF exceeds the worker response limit")
        return PagePdfResponse(
            type="page_pdf_result",
            request_id=request_id,
            page_number=request.page_number,
            width_points=width_points,
            height_points=height_points,
            data_base64=base64.b64encode(pdf_bytes).decode("ascii"),
        )
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return ParseWorkerResponse(
            type="error",
            request_id=request_id,
            error_code="PAGE_EXPORT_FAILED",
            error_type=type(exc).__name__,
        )


def _handle_image_ocr(
    raw_request: bytes,
    parsers: tuple[FastTextParser, StructuredParser, StructuredParser, OcrParser],
) -> ImageOcrResponse | ParseWorkerResponse:
    request_id = "0" * 32
    try:
        request = ImageOcrRequest.model_validate_json(raw_request)
        request_id = request.request_id
        if len(request.images) > MAX_OCR_BATCH_IMAGES:
            raise ValueError("OCR image count exceeds the worker limit")
        encoded_size = sum(len(item.data_base64) for item in request.images)
        if encoded_size > MAX_OCR_BATCH_BYTES:
            raise ValueError("OCR image batch exceeds the worker size limit")
        results = []
        with TemporaryDirectory(prefix="image-ocr-") as directory:
            root = Path(directory)
            for index, item in enumerate(request.images):
                try:
                    data = base64.b64decode(item.data_base64, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise ValueError("OCR image is not valid base64") from exc
                if not data or len(data) > MAX_OCR_IMAGE_BYTES:
                    raise ValueError("OCR image is outside the worker size limit")
                if (
                    not data.startswith(b"\x89PNG\r\n\x1a\n")
                    or len(data) < 24
                    or data[12:16] != b"IHDR"
                ):
                    raise ValueError("OCR input must be a valid PNG crop")
                width = int.from_bytes(data[16:20], "big")
                height = int.from_bytes(data[20:24], "big")
                if width <= 0 or height <= 0 or width * height > 4_000_000:
                    raise ValueError("OCR image exceeds the pixel limit")
                pixmap = pymupdf.Pixmap(data)
                if pixmap.width != width or pixmap.height != height:
                    raise ValueError("PNG dimensions are inconsistent")
                path = root / f"{index}.png"
                path.write_bytes(data)
                lines = parsers[3]._tesseract(path)
                results.append(
                    ImageOcrText(
                        image_id=item.image_id,
                        text="\n".join(line[0] for line in lines),
                    )
                )
        return ImageOcrResponse(
            type="ocr_result", request_id=request_id, results=tuple(results)
        )
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return ParseWorkerResponse(
            type="error",
            request_id=request_id,
            error_code="IMAGE_OCR_FAILED",
            error_type=type(exc).__name__,
        )
def main() -> int:
    try:
        parsers = _initialize_parsers()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 2

    output = sys.stdout.buffer
    _write_message(output, ParseWorkerResponse(type="ready"))
    input_stream = sys.stdin.buffer
    try:
        while raw_request := _read_message(input_stream):
            response = _handle(raw_request, parsers)
            _write_message(output, response)
    except (EOFError, OSError, ValueError):
        traceback.print_exc(file=sys.stderr)
        return 3
    finally:
        for parser in parsers[1:3]:
            parser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
