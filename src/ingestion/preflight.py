import hashlib
from pathlib import Path

import pymupdf

from src.ingestion.contracts import DocumentFeatures, PageFeatures


class PdfPreflightError(Exception):
    """Base error for failures while inspecting a PDF."""


class PdfNotFoundError(PdfPreflightError, FileNotFoundError):
    """Raised when the requested PDF path does not exist."""


class InvalidPdfError(PdfPreflightError):
    """Raised when the file is not a readable, non-empty PDF."""


def inspect_pdf(path: Path) -> DocumentFeatures:
    try:
        with path.open("rb") as source:
            file_size_bytes = 0
            content_hash = hashlib.sha256()
            while chunk := source.read(1024 * 1024):
                file_size_bytes += len(chunk)
                content_hash.update(chunk)
    except FileNotFoundError as exc:
        raise PdfNotFoundError(f"PDF file not found: {path}") from exc

    try:
        with pymupdf.open(path) as document:
            if not document.is_pdf:
                raise InvalidPdfError(f"File is not a PDF: {path}")
            if document.needs_pass:
                raise InvalidPdfError(f"PDF is password-protected: {path}")

            page_count = document.page_count
            if page_count == 0:
                raise InvalidPdfError(f"PDF contains no pages: {path}")

            pages = tuple(
                PageFeatures(
                    page_number=page_number,
                    text_char_count=len(page.get_text()),
                    image_count=len(page.get_image_info(xrefs=True)),
                    drawing_count=len(page.get_drawings()),
                )
                for page_number, page in enumerate(document, start=1)
            )
    except pymupdf.FileNotFoundError as exc:
        raise PdfNotFoundError(f"PDF file not found: {path}") from exc
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise InvalidPdfError(f"Unable to read PDF file: {path}") from exc

    return DocumentFeatures(
        content_sha256=content_hash.hexdigest(),
        file_size_bytes=file_size_bytes,
        page_count=page_count,
        pages=pages,
    )
