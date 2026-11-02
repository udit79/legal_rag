import csv
import io
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol, cast

import pymupdf

from src.ingestion.parser_contracts import (
    CanonicalBlock,
    CanonicalPage,
    SourceCoordinates,
)


class PageParser(Protocol):
    def parse_page(self, source: Path, page_number: int) -> CanonicalPage: ...


class DoclingDocument(Protocol):
    def iterate_items(self) -> Iterable[tuple[object, object]]: ...


class DoclingResult(Protocol):
    status: object
    errors: object
    document: DoclingDocument


class DoclingConverter(Protocol):
    def convert(
        self,
        source: Path,
        *,
        page_range: tuple[int, int],
        raises_on_error: bool,
    ) -> DoclingResult: ...


class RapidOcrEngine(Protocol):
    def __call__(self, image_path: str) -> object: ...


def _coordinates(
    bounds: tuple[float, float, float, float],
    *,
    unit: Literal["PDF_POINT", "PIXEL"],
    origin: Literal["TOP_LEFT", "BOTTOM_LEFT"],
) -> SourceCoordinates:
    x0, y0, x1, y1 = bounds
    return SourceCoordinates(
        x0=min(x0, x1),
        y0=min(y0, y1),
        x1=max(x0, x1),
        y1=max(y0, y1),
        unit=unit,
        origin=origin,
    )


def _page_status(
    blocks: tuple[CanonicalBlock, ...],
) -> Literal["PARSED", "BLANK", "VECTOR_ONLY"]:
    if not blocks:
        return "BLANK"
    if all(block.block_type == "VECTOR" for block in blocks):
        return "VECTOR_ONLY"
    return "PARSED"


def _check_page_content_streams(
    document: pymupdf.Document, page: pymupdf.Page
) -> None:
    for xref in page.get_contents():
        try:
            stream = document.xref_stream(xref)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Unable to read page content stream {xref}") from exc
        if stream is None:
            raise ValueError(f"Page content stream {xref} is missing")
        if not stream:
            try:
                stream_object = document.xref_object(xref)
            except (RuntimeError, ValueError) as exc:
                raise ValueError(f"Unable to inspect page content stream {xref}") from exc
            if "/Filter" in stream_object:
                raise ValueError(
                    f"Page content stream {xref} could not be decoded"
                )


class FastTextParser:
    """Extract native text and visual primitives with PyMuPDF."""

    parser_name = "FAST_TEXT"

    def parse_page(self, source: Path, page_number: int) -> CanonicalPage:
        with pymupdf.open(source) as document:
            if document.needs_pass:
                raise PermissionError("PDF is encrypted and requires a password")
            if page_number < 1 or page_number > document.page_count:
                raise IndexError(f"Page number {page_number} is outside the PDF")

            page = document.load_page(page_number - 1)
            _check_page_content_streams(document, page)
            text_data = page.get_text("dict", sort=False)
            if not isinstance(text_data, dict):
                raise TypeError("PyMuPDF did not return a text dictionary")
            raw_blocks = text_data.get("blocks", [])
            if not isinstance(raw_blocks, list):
                raise TypeError("PyMuPDF text dictionary has an invalid blocks field")
            blocks: list[CanonicalBlock] = []

            for raw_block in raw_blocks:
                if not isinstance(raw_block, dict):
                    raise TypeError("PyMuPDF returned an invalid text block")
                bbox = raw_block.get("bbox")
                if (
                    not isinstance(bbox, (tuple, list))
                    or len(bbox) != 4
                ):
                    continue
                block_type = raw_block.get("type")
                if block_type == 0:
                    text = "".join(
                        span.get("text", "")
                        for line in raw_block.get("lines", [])
                        for span in line.get("spans", [])
                    )
                    if not text.strip():
                        continue
                    kind = "TEXT"
                elif block_type == 1:
                    text = ""
                    kind = "IMAGE"
                else:
                    continue

                blocks.append(
                    CanonicalBlock(
                        page_number=page_number,
                        text=text,
                        block_type=kind,
                        reading_order=len(blocks),
                        source_coordinates=_coordinates(
                            (
                                float(bbox[0]),
                                float(bbox[1]),
                                float(bbox[2]),
                                float(bbox[3]),
                            ),
                            unit="PDF_POINT",
                            origin="TOP_LEFT",
                        ),
                    )
                )

            for drawing in page.get_drawings():
                rect = drawing.get("rect")
                if rect is None or rect.is_empty:
                    continue
                blocks.append(
                    CanonicalBlock(
                        page_number=page_number,
                        text="",
                        block_type="VECTOR",
                        reading_order=len(blocks),
                        source_coordinates=_coordinates(
                            (rect.x0, rect.y0, rect.x1, rect.y1),
                            unit="PDF_POINT",
                            origin="TOP_LEFT",
                        ),
                    )
                )

        canonical_blocks = tuple(blocks)
        return CanonicalPage(
            page_number=page_number,
            parser=self.parser_name,
            status=_page_status(canonical_blocks),
            blocks=canonical_blocks,
        )


class StructuredParser:
    """Extract layout-aware text and tables with Docling, one page at a time."""

    parser_name = "DOCLING"

    def __init__(
        self, converter: DoclingConverter | None = None, *, do_ocr: bool = False
    ) -> None:
        self._converter = converter
        self.do_ocr = do_ocr

    def _get_converter(self) -> DoclingConverter:
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                RapidOcrOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            if self.do_ocr:
                options = PdfPipelineOptions(
                    do_ocr=True,
                    ocr_options=RapidOcrOptions(),
                    do_table_structure=True,
                )
            else:
                options = PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=True,
                )
            self._converter = cast(
                DoclingConverter,
                DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
                    }
                ),
            )
        return self._converter

    def parse_page(self, source: Path, page_number: int) -> CanonicalPage:
        converter = self._get_converter()
        result = converter.convert(
            source,
            page_range=(page_number, page_number),
            raises_on_error=False,
        )
        status = result.status
        status_name = str(
            getattr(
                status,
                "name",
                getattr(status, "value", status),
            )
        ).upper()
        if status_name not in {"SUCCESS", "PARTIAL_SUCCESS"}:
            raise RuntimeError(
                f"Docling failed on page {page_number}: "
                f"{result.errors}"
            )

        document = result.document
        blocks: list[CanonicalBlock] = []
        for item, _ in document.iterate_items():
            provenances = getattr(item, "prov", None) or ()
            page_provenance = next(
                (
                    provenance
                    for provenance in provenances
                    if getattr(provenance, "page_no", None) == page_number
                ),
                None,
            )
            if not provenances:
                raise RuntimeError(
                    "Docling returned an item without source page provenance"
                )
            if page_provenance is None:
                raise RuntimeError(
                    "Docling returned an item without provenance for the "
                    f"requested page {page_number}"
                )

            raw_label = getattr(item, "label", "")
            label = str(getattr(raw_label, "value", raw_label)).casefold()
            if "table" in label:
                export_table = getattr(item, "export_to_markdown")
                text = export_table(doc=document)
                block_type = "TABLE"
            else:
                value = getattr(item, "text", None)
                text = value if isinstance(value, str) else ""
                if "picture" in label or "image" in label:
                    block_type = "IMAGE"
                elif not text.strip():
                    continue
                elif label in {
                    "text",
                    "paragraph",
                    "section_header",
                    "title",
                    "caption",
                }:
                    block_type = "TEXT"
                else:
                    block_type = "OTHER"

            bbox = getattr(page_provenance, "bbox")
            source_coordinates = _coordinates(
                (
                    float(getattr(bbox, "l")),
                    float(getattr(bbox, "t")),
                    float(getattr(bbox, "r")),
                    float(getattr(bbox, "b")),
                ),
                unit="PDF_POINT",
                origin="BOTTOM_LEFT",
            )
            blocks.append(
                CanonicalBlock(
                    page_number=page_number,
                    text=text,
                    block_type=block_type,
                    reading_order=len(blocks),
                    source_coordinates=source_coordinates,
                )
            )

        canonical_blocks = tuple(blocks)
        if not canonical_blocks:
            visual_page = FastTextParser().parse_page(source, page_number)
            return visual_page.model_copy(update={"parser": "DOCLING+PYMUPDF"})
        return CanonicalPage(
            page_number=page_number,
            parser=self.parser_name,
            status=_page_status(canonical_blocks),
            blocks=canonical_blocks,
        )


class OcrParser:
    """OCR a page using Tesseract or the optional RapidOCR engine."""

    def __init__(
        self,
        engine: Literal["tesseract", "rapidocr"] = "tesseract",
        *,
        tesseract_command: str | None = None,
        timeout_seconds: int = 120,
        rapidocr_engine: RapidOcrEngine | None = None,
    ) -> None:
        if engine not in {"tesseract", "rapidocr"}:
            raise ValueError(f"Unsupported OCR engine: {engine}")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        self.engine = engine
        self.tesseract_command = tesseract_command
        self.timeout_seconds = timeout_seconds
        self._rapidocr_engine = rapidocr_engine

    @property
    def parser_name(self) -> str:
        return "TESSERACT" if self.engine == "tesseract" else "RAPIDOCR"

    def _get_rapidocr(self) -> RapidOcrEngine:
        if self._rapidocr_engine is None:
            from rapidocr import RapidOCR

            self._rapidocr_engine = cast(RapidOcrEngine, RapidOCR())
        return self._rapidocr_engine

    def _tesseract(
        self, image_path: Path
    ) -> list[tuple[str, tuple[float, ...], float | None]]:
        command = self.tesseract_command or os.environ.get("TESSERACT_CMD")
        command = command or shutil.which("tesseract")
        if not command:
            raise FileNotFoundError(
                "Tesseract executable was not found; install it, add it to PATH, "
                "or set TESSERACT_CMD"
            )
        result = subprocess.run(
            [
                command,
                str(image_path),
                "stdout",
                "-l",
                "eng",
                "--oem",
                "1",
                "--psm",
                "6",
                "tsv",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Tesseract failed with exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )

        lines: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in csv.DictReader(io.StringIO(result.stdout), delimiter="\t"):
            text = row.get("text", "").strip()
            if text:
                key = (row["block_num"], row["par_num"], row["line_num"])
                lines.setdefault(key, []).append(row)

        output = []
        for words in lines.values():
            text = " ".join(word["text"].strip() for word in words)
            left = min(float(word["left"]) for word in words)
            top = min(float(word["top"]) for word in words)
            right = max(float(word["left"]) + float(word["width"]) for word in words)
            bottom = max(float(word["top"]) + float(word["height"]) for word in words)
            confidences = [
                float(word["conf"]) for word in words if float(word["conf"]) >= 0
            ]
            confidence = (
                sum(confidences) / len(confidences) if confidences else None
            )
            output.append((text, (left, top, right, bottom), confidence))
        return output

    def _rapidocr(
        self, image_path: Path
    ) -> list[tuple[str, tuple[float, ...], float | None]]:
        result = self._get_rapidocr()(str(image_path))
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if boxes is None and texts is None and scores is None:
            return []
        if boxes is None or texts is None:
            raise RuntimeError("RapidOCR returned incomplete text detection results")
        if scores is not None and len(scores) != len(texts):
            raise RuntimeError("RapidOCR returned a different number of scores and texts")

        output = []
        for index, (box, text) in enumerate(zip(boxes, texts, strict=True)):
            if not isinstance(text, str) or not text.strip():
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            confidence = float(scores[index]) * 100 if scores is not None else None
            output.append(
                (
                    text.strip(),
                    (min(xs), min(ys), max(xs), max(ys)),
                    confidence,
                )
            )
        return output

    def parse_page(self, source: Path, page_number: int) -> CanonicalPage:
        with pymupdf.open(source) as document:
            if document.needs_pass:
                raise PermissionError("PDF is encrypted and requires a password")
            if page_number < 1 or page_number > document.page_count:
                raise IndexError(f"Page number {page_number} is outside the PDF")
            page = document.load_page(page_number - 1)
            _check_page_content_streams(document, page)
            pixmap = page.get_pixmap(
                dpi=300, colorspace=pymupdf.csRGB, alpha=False
            )
            page_width = page.rect.width
            page_height = page.rect.height
            scale_x = page_width / pixmap.width
            scale_y = page_height / pixmap.height
            visual_page = FastTextParser().parse_page(source, page_number)

            with tempfile.TemporaryDirectory(prefix="legal-rag-ocr-") as directory:
                image_path = Path(directory) / f"page-{page_number}.png"
                pixmap.save(image_path)
                if self.engine == "tesseract":
                    recognized = self._tesseract(image_path)
                else:
                    recognized = self._rapidocr(image_path)

        blocks = [
            block
            for block in visual_page.blocks
            if block.block_type in {"TEXT", "IMAGE", "VECTOR"}
        ]
        for text, bbox, confidence in recognized:
            blocks.append(
                CanonicalBlock(
                    page_number=page_number,
                    text=text,
                    block_type="TEXT",
                    reading_order=len(blocks),
                    source_coordinates=_coordinates(
                        (
                            bbox[0] * scale_x,
                            bbox[1] * scale_y,
                            bbox[2] * scale_x,
                            bbox[3] * scale_y,
                        ),
                        unit="PDF_POINT",
                        origin="TOP_LEFT",
                    ),
                    confidence=confidence,
                )
            )

        blocks.sort(
            key=lambda block: (
            block.source_coordinates.y0,
            block.source_coordinates.x0,
            block.reading_order,
            )
        )
        ordered_blocks = tuple(
            block.model_copy(update={"reading_order": index})
            for index, block in enumerate(blocks)
        )
        return CanonicalPage(
            page_number=page_number,
            parser=self.parser_name,
            status=_page_status(ordered_blocks),
            blocks=ordered_blocks,
        )
