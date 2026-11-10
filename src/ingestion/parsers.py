from __future__ import annotations
import csv
import gc
import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
import warnings
from contextlib import contextmanager
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast
import pymupdf
from src.ingestion.parser_contracts import CanonicalBlock,CanonicalPage,ParserFailure,SourceCoordinates

if TYPE_CHECKING:
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
    from docling_core.types.doc.items.node import NodeItem

OCR_DUPLICATE_OVERLAP_THRESHOLD = 0.8
NATIVE_WORD_GAP_EM_THRESHOLD = 0.25
_COVER_APPEAL_NUMBER = re.compile(r"\bcivil\s+appeal\s+no\.", re.IGNORECASE)
_COVER_SUPREME_COURT = re.compile(r"\bsupreme\s+court\s+of\s+india\b", re.IGNORECASE)
_COVER_VERSUS = re.compile(r"\bversus\b", re.IGNORECASE)
_COVER_APPELLANT = re.compile(r"\bappellant\s*\(s\)", re.IGNORECASE)
_COVER_RESPONDENT = re.compile(r"\brespondent\s*\(s\)", re.IGNORECASE)

@dataclass
class _VisualRow:
    top: float
    bottom: float
    blocks: list[CanonicalBlock]

    @property
    def center(self) -> float:
        return (self.top + self.bottom) / 2

class PageParser(Protocol):
    def parse_page(self, source: Path, page_number: int) -> CanonicalPage: ...

class DoclingDocument(Protocol):
    def iterate_items(self) -> Iterable[tuple[NodeItem, int]]: ...

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

def _repair_multi_appeal_cover_order(
    blocks: tuple[CanonicalBlock, ...],
) -> tuple[CanonicalBlock, ...]:
    """Repair only a confidently detected, geometrically readable appeal cover."""
    texts = [block.text for block in blocks]
    if not (
        any(_COVER_SUPREME_COURT.search(text) for text in texts)
        and sum(bool(_COVER_APPEAL_NUMBER.search(text)) for text in texts) >= 2
        and sum(bool(_COVER_VERSUS.search(text)) for text in texts) >= 2
        and sum(bool(_COVER_APPELLANT.search(text)) for text in texts) >= 2
        and sum(bool(_COVER_RESPONDENT.search(text)) for text in texts) >= 2
    ):
        return blocks
    coordinates = [block.source_coordinates for block in blocks]
    if any(
        coordinate.unit != "PDF_POINT"
        or coordinate.origin != coordinates[0].origin
        or coordinate.x1 <= coordinate.x0
        or coordinate.y1 <= coordinate.y0
        for coordinate in coordinates
    ):
        return blocks

    # Convert both supported coordinate origins into a top-left y interval.
    rows: list[_VisualRow] = []
    positioned = []
    for block in blocks:
        box = block.source_coordinates
        top, bottom = (
            (box.y0, box.y1) if box.origin == "TOP_LEFT" else (-box.y1, -box.y0)
        )
        positioned.append((block, top, bottom))
    positioned.sort(key=lambda item: ((item[1] + item[2]) / 2, item[0].reading_order))

    for block, top, bottom in positioned:
        row = next(
            (
                candidate
                for candidate in rows
                if min(bottom, candidate.bottom) - max(top, candidate.top)
                >= 0.5 * min(bottom - top, candidate.bottom - candidate.top)
                or abs((top + bottom) / 2 - candidate.center) <= 3.0
            ),
            None,
        )
        if row is None:
            rows.append(_VisualRow(top=top, bottom=bottom, blocks=[block]))
        else:
            row.top = min(row.top, top)
            row.bottom = max(row.bottom, bottom)
            row.blocks.append(block)

    ordered_rows = sorted(rows, key=lambda row: (row.top, row.center))
    ordered = [
        block
        for row in ordered_rows
        for block in sorted(
            row.blocks,
            key=lambda block: (block.source_coordinates.x0, block.reading_order),
        )
    ]
    if len(ordered) != len(blocks) or len({block.block_id for block in ordered}) != len(
        blocks
    ):
        return blocks
    return tuple(
        block.model_copy(update={"reading_order": index})
        for index, block in enumerate(ordered)
    )

def _normalize_text_line(text: str) -> str:
    return " ".join(text.casefold().split())

def _overlap_relative_to_smaller(first: pymupdf.Rect, second: pymupdf.Rect) -> float:
    intersection = first & second
    intersection_area = max(0.0, intersection.width) * max(0.0, intersection.height)
    smaller_area = min(
        max(0.0, first.width) * max(0.0, first.height),
        max(0.0, second.width) * max(0.0, second.height),
    )
    if smaller_area == 0:
        return 0.0
    return intersection_area / smaller_area

def _span_gap_indicates_word_boundary(
    previous_span: dict[str, object],
    current_span: dict[str, object],
    direction: tuple[float, float],
) -> bool:
    previous_bbox = previous_span.get("bbox")
    current_bbox = current_span.get("bbox")
    previous_size = previous_span.get("size")
    current_size = current_span.get("size")
    if (
        not isinstance(previous_bbox, (tuple, list))
        or len(previous_bbox) != 4
        or not isinstance(current_bbox, (tuple, list))
        or len(current_bbox) != 4
        or not isinstance(previous_size, (int, float))
        or not isinstance(current_size, (int, float))
    ):
        return False

    if abs(direction[0]) >= abs(direction[1]):
        gap = (
            float(current_bbox[0]) - float(previous_bbox[2])
            if direction[0] >= 0
            else float(previous_bbox[0]) - float(current_bbox[2])
        )
    else:
        gap = (
            float(current_bbox[1]) - float(previous_bbox[3])
            if direction[1] >= 0
            else float(previous_bbox[1]) - float(current_bbox[3])
        )
    word_gap_threshold = (
        min(float(previous_size), float(current_size)) * NATIVE_WORD_GAP_EM_THRESHOLD
    )
    return gap >= word_gap_threshold

def _join_native_spans(
    spans: list[dict[str, object]],
    direction: tuple[float, float],
) -> str:
    text = ""
    previous_span: dict[str, object] | None = None
    for span in spans:
        span_text = span.get("text", "")
        if not isinstance(span_text, str):
            continue
        if (
            previous_span is not None
            and text
            and span_text
            and not text[-1].isspace()
            and not span_text[0].isspace()
            and _span_gap_indicates_word_boundary(previous_span, span, direction)
        ):
            text += " "
        text += span_text
        previous_span = span
    return text

def _native_text_line_geometry(
    page: pymupdf.Page,
) -> list[tuple[str, pymupdf.Rect]]:
    text_data = page.get_text("dict", sort=False)
    if not isinstance(text_data, dict):
        raise TypeError("PyMuPDF did not return a text dictionary")
    raw_blocks = text_data.get("blocks", [])
    if not isinstance(raw_blocks, list):
        raise TypeError("PyMuPDF text dictionary has an invalid blocks field")

    lines_with_geometry = []
    for raw_block in raw_blocks:
        if not isinstance(raw_block, dict) or raw_block.get("type") != 0:
            continue
        raw_lines = raw_block.get("lines", [])
        if not isinstance(raw_lines, list):
            raise TypeError("PyMuPDF text block has an invalid lines field")
        for line in raw_lines:
            if not isinstance(line, dict):
                raise TypeError("PyMuPDF returned an invalid text line")
            spans = line.get("spans", [])
            if not isinstance(spans, list):
                raise TypeError("PyMuPDF text line has an invalid spans field")
            raw_direction = line.get("dir", (1.0, 0.0))
            if not isinstance(raw_direction, (tuple, list)) or len(raw_direction) != 2:
                raise TypeError("PyMuPDF text line has an invalid direction")
            direction = (float(raw_direction[0]), float(raw_direction[1]))
            text = _join_native_spans(spans, direction)
            normalized = _normalize_text_line(text)
            bbox = line.get("bbox")
            if not normalized or not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                continue
            lines_with_geometry.append(
                (
                    normalized,
                    pymupdf.Rect(*(float(value) for value in bbox)),
                )
            )
    return lines_with_geometry

def _check_page_content_streams(document: pymupdf.Document, page: pymupdf.Page) -> None:
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
                raise ValueError(
                    f"Unable to inspect page content stream {xref}"
                ) from exc
            if "/Filter" in stream_object:
                raise ValueError(f"Page content stream {xref} could not be decoded")

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
                if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                    continue
                block_type = raw_block.get("type")
                if block_type == 0:
                    raw_lines = raw_block.get("lines", [])
                    if not isinstance(raw_lines, list):
                        raise TypeError("PyMuPDF text block has an invalid lines field")
                    line_texts = []
                    for line in raw_lines:
                        if not isinstance(line, dict):
                            raise TypeError("PyMuPDF returned an invalid text line")
                        spans = line.get("spans", [])
                        if not isinstance(spans, list) or not all(
                            isinstance(span, dict) for span in spans
                        ):
                            raise TypeError("PyMuPDF text line has invalid spans")
                        raw_direction = line.get("dir", (1.0, 0.0))
                        if (
                            not isinstance(raw_direction, (tuple, list))
                            or len(raw_direction) != 2
                        ):
                            raise TypeError(
                                "PyMuPDF text line has an invalid direction"
                            )
                        direction = (
                            float(raw_direction[0]),
                            float(raw_direction[1]),
                        )
                        line_texts.append(_join_native_spans(spans, direction))
                    text = "\n".join(line_texts)
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

def _captionless_table_serializer(
    document: DoclingDocument,
) -> MarkdownDocSerializer:
    from docling_core.transforms.serializer.markdown import (
        MarkdownDocSerializer,
        MarkdownParams,
    )
    from docling_core.types.doc.labels import DocItemLabel
    from docling_core.types.doc.document import DoclingDocument as CoreDoclingDocument

    default_params = MarkdownParams()
    params = MarkdownParams(labels=default_params.labels - {DocItemLabel.CAPTION})
    return MarkdownDocSerializer(doc=cast(CoreDoclingDocument, document), params=params)

@contextmanager
def _docling_deprecation_compat() -> Generator[None, None, None]:
    # Docling 2.145 accesses these deprecated compatibility fields internally.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="^deprecated$",
            category=DeprecationWarning,
            module=r"docling\.models\.stages\.ocr\.rapid_ocr_model",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^`force_full_page_ocr` is deprecated; set `mode=OcrMode\.FULL_PAGE` instead\.$",
            category=DeprecationWarning,
            module=r"pydantic\.main",
        )
        yield

def _recover_native_text_under_empty_images(
    source: Path,
    page_number: int,
    blocks: tuple[CanonicalBlock, ...],
) -> tuple[CanonicalBlock, ...]:
    empty_images = [
        block
        for block in blocks
        if block.block_type == "IMAGE"
        and not block.text.strip()
        and block.source_coordinates.unit == "PDF_POINT"
    ]
    if not empty_images:
        return ()

    represented_text = " ".join(
        _normalize_text_line(block.text) for block in blocks if block.text.strip()
    )
    recovered: list[CanonicalBlock] = []
    with pymupdf.open(source) as document:
        if document.needs_pass:
            return ()
        if page_number < 1 or page_number > document.page_count:
            return ()
        page = document.load_page(page_number - 1)
        page_height = page.rect.height
        image_regions = [
            pymupdf.Rect(
                block.source_coordinates.x0,
                (
                    page_height - block.source_coordinates.y1
                    if block.source_coordinates.origin == "BOTTOM_LEFT"
                    else block.source_coordinates.y0
                ),
                block.source_coordinates.x1,
                (
                    page_height - block.source_coordinates.y0
                    if block.source_coordinates.origin == "BOTTOM_LEFT"
                    else block.source_coordinates.y1
                ),
            )
            for block in empty_images
        ]
        _check_page_content_streams(document, page)
        text_data = page.get_text("dict", sort=False)
        if not isinstance(text_data, dict):
            return ()
        raw_blocks_value = text_data.get("blocks", [])
        if not isinstance(raw_blocks_value, list):
            return ()

        def add_run(run: list[tuple[str, pymupdf.Rect]]) -> None:
            if not run:
                return
            text = "\n".join(line_text for line_text, _ in run)
            normalized = _normalize_text_line(text)
            if not normalized or normalized in represented_text:
                return
            x0 = min(rect.x0 for _, rect in run)
            y0 = min(rect.y0 for _, rect in run)
            x1 = max(rect.x1 for _, rect in run)
            y1 = max(rect.y1 for _, rect in run)
            recovered.append(
                CanonicalBlock(
                    page_number=page_number,
                    text=text,
                    block_type="TEXT",
                    reading_order=len(blocks) + len(recovered),
                    source_coordinates=_coordinates(
                        (x0, page.rect.height - y1, x1, page.rect.height - y0),
                        unit="PDF_POINT",
                        origin="BOTTOM_LEFT",
                    ),
                )
            )

        for raw_block_value in raw_blocks_value:
            if not isinstance(raw_block_value, dict):
                continue
            raw_block = cast(dict[str, object], raw_block_value)
            if raw_block.get("type") != 0:
                continue
            raw_lines = raw_block.get("lines", [])
            if not isinstance(raw_lines, list):
                continue
            run: list[tuple[str, pymupdf.Rect]] = []
            for raw_line in raw_lines:
                if not isinstance(raw_line, dict):
                    add_run(run)
                    run = []
                    continue
                line = cast(dict[str, object], raw_line)
                spans = line.get("spans", [])
                direction = line.get("dir", (1.0, 0.0))
                bbox = line.get("bbox")
                if (
                    not isinstance(spans, list)
                    or not all(isinstance(span, dict) for span in spans)
                    or not isinstance(direction, (tuple, list))
                    or len(direction) != 2
                    or not isinstance(bbox, (tuple, list))
                    or len(bbox) != 4
                ):
                    add_run(run)
                    run = []
                    continue
                text = _join_native_spans(
                    spans, (float(direction[0]), float(direction[1]))
                )
                rect = pymupdf.Rect(*(float(value) for value in bbox))
                overlaps_image = any(
                    _overlap_relative_to_smaller(rect, region) >= 0.8
                    for region in image_regions
                )
                if text.strip() and overlaps_image:
                    run.append((text, rect))
                else:
                    add_run(run)
                    run = []
            add_run(run)

    return tuple(recovered)

class StructuredParser:
    """Extract layout-aware text and tables with Docling, one page at a time."""

    parser_name = "DOCLING"

    def __init__(
        self, converter: DoclingConverter | None = None, *, do_ocr: bool = False
    ) -> None:
        self._converter = converter
        self.do_ocr = do_ocr
        self._converter_lock = threading.RLock()

    def initialize(self) -> None:
        """Load the Docling PDF pipeline before the parser serves requests."""
        from docling.datamodel.base_models import InputFormat

        with self._converter_lock:
            converter = self._get_converter()
            initialize_pipeline = getattr(converter, "initialize_pipeline", None)
            if callable(initialize_pipeline):
                initialize_pipeline(InputFormat.PDF)

    def close(self) -> None:
        """Release the cached converter and its initialized model pipeline."""
        with self._converter_lock:
            self._converter = None
            gc.collect()

    def _get_converter(self) -> DoclingConverter:
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                RapidOcrOptions,
            )
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.document_converter import DocumentConverter, PdfFormatOption

            if self.do_ocr:
                options = PdfPipelineOptions(
                    do_ocr=True,
                    ocr_options=RapidOcrOptions(),
                    do_table_structure=True,
                    generate_page_images=True,
                )
            else:
                options = PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=True,
                    generate_page_images=True,
                )
            self._converter = cast(
                DoclingConverter,
                DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=options,
                            backend=PyPdfiumDocumentBackend,
                        )
                    }
                ),
            )
        return self._converter

    def parse_page(self, source: Path, page_number: int) -> CanonicalPage:
        with self._converter_lock:
            converter = self._get_converter()
            with _docling_deprecation_compat():
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
            raise RuntimeError(f"Docling failed on page {page_number}: {result.errors}")

        document = result.document
        blocks: list[CanonicalBlock] = []
        table_serializer: MarkdownDocSerializer | None = None
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
                if table_serializer is None:
                    table_serializer = _captionless_table_serializer(document)
                text = table_serializer.serialize(item=item).text
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
            coord_origin = getattr(bbox, "coord_origin", None)
            origin_name = getattr(coord_origin, "value", coord_origin)
            if origin_name == "TOPLEFT":
                origin = "TOP_LEFT"
            elif origin_name == "BOTTOMLEFT":
                origin = "BOTTOM_LEFT"
            else:
                raise ValueError(
                    f"Docling returned unsupported coordinate origin: {coord_origin!r}"
                )
            source_coordinates = _coordinates(
                (
                    float(getattr(bbox, "l")),
                    float(getattr(bbox, "t")),
                    float(getattr(bbox, "r")),
                    float(getattr(bbox, "b")),
                ),
                unit="PDF_POINT",
                origin=origin,
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
        recovered_blocks = _recover_native_text_under_empty_images(
            source, page_number, canonical_blocks
        )
        warnings = ()
        if recovered_blocks:
            canonical_blocks += recovered_blocks
            warnings = (
                ParserFailure(
                    parser="PYMUPDF",
                    code="NATIVE_TEXT_RECOVERED",
                    message=(
                        f"Recovered {len(recovered_blocks)} native text block(s) "
                        "from regions overlapping empty Docling image blocks. "
                        "Recovered blocks are inserted into visual row order for "
                        "recognized multi-appeal covers."
                    ),
                ),
            )
        canonical_blocks = _repair_multi_appeal_cover_order(canonical_blocks)
        if not canonical_blocks:
            visual_page = FastTextParser().parse_page(source, page_number)
            return visual_page.model_copy(update={"parser": "DOCLING+PYMUPDF"})
        return CanonicalPage(
            page_number=page_number,
            parser=self.parser_name,
            status=_page_status(canonical_blocks),
            blocks=canonical_blocks,
            warnings=warnings,
        )

class OcrParser:
    """OCR a page using Tesseract or the optional RapidOCR engine."""

    def __init__(
        self,
        engine: Literal["tesseract", "rapidocr"] = "tesseract",
        *,
        tesseract_command: str | None = None,
        tesseract_language: str | None = None,
        page_segmentation_mode: int | None = None,
        timeout_seconds: int = 120,
        rapidocr_engine: RapidOcrEngine | None = None,
    ) -> None:
        if engine not in {"tesseract", "rapidocr"}:
            raise ValueError(f"Unsupported OCR engine: {engine}")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        language = tesseract_language or os.environ.get("TESSERACT_LANG", "eng")
        if not re.fullmatch(r"[A-Za-z0-9_+.-]+", language):
            raise ValueError("Tesseract language must be a language code or joined codes")
        psm = page_segmentation_mode
        if psm is None:
            configured_psm = os.environ.get("TESSERACT_PSM", "3")
            try:
                psm = int(configured_psm)
            except ValueError as exc:
                raise ValueError("TESSERACT_PSM must be an integer from 0 to 13") from exc
        if type(psm) is not int or not 0 <= psm <= 13:
            raise ValueError("Tesseract page segmentation mode must be from 0 to 13")
        self.engine = engine
        self.tesseract_command = tesseract_command
        self.tesseract_language = language
        self.page_segmentation_mode = psm
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
                self.tesseract_language,
                "--oem",
                "1",
                "--psm",
                str(self.page_segmentation_mode),
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
            confidence = sum(confidences) / len(confidences) if confidences else None
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
            raise RuntimeError(
                "RapidOCR returned a different number of scores and texts"
            )

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
            pixmap = page.get_pixmap(dpi=300, colorspace=pymupdf.csRGB, alpha=False)
            page_width = page.rect.width
            page_height = page.rect.height
            scale_x = page_width / pixmap.width
            scale_y = page_height / pixmap.height
            native_text_geometry = _native_text_line_geometry(page)
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
        seen_text_lines = native_text_geometry
        for text, bbox, confidence in recognized:
            ocr_rect = pymupdf.Rect(
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y,
            )
            retained_lines = []
            for line in text.splitlines():
                normalized_line = _normalize_text_line(line)
                if not normalized_line:
                    continue
                is_duplicate = any(
                    normalized_line == existing_text
                    and _overlap_relative_to_smaller(ocr_rect, existing_rect)
                    >= OCR_DUPLICATE_OVERLAP_THRESHOLD
                    for existing_text, existing_rect in seen_text_lines
                )
                if is_duplicate:
                    continue
                seen_text_lines.append((normalized_line, ocr_rect))
                retained_lines.append(line.strip())
            if not retained_lines:
                continue
            blocks.append(
                CanonicalBlock(
                    page_number=page_number,
                    text="\n".join(retained_lines),
                    block_type="TEXT",
                    reading_order=len(blocks),
                    source_coordinates=_coordinates(
                        (
                            ocr_rect.x0,
                            ocr_rect.y0,
                            ocr_rect.x1,
                            ocr_rect.y1,
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
