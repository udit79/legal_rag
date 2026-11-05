"""OCR and screen preserved image crops without altering the source image."""

from __future__ import annotations

import base64
import hashlib
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from src.ingestion.pipeline import UnrecognizedImageCrop
from src.ingestion.parser_contracts import SourceCoordinates
from src.security.ingestion_screening.contracts import ScreeningFlag
from src.security.ingestion_screening.enums import AdmissionDecision
from src.security.ingestion_screening.screening_service import ScreeningService


class ImageOcr(Protocol):
    async def ocr_image_crops(
        self, crops: tuple[UnrecognizedImageCrop, ...]
    ) -> dict[str, str]: ...


class ImageScreeningResult(BaseModel):
    """Security sidecar; flag offsets refer to OCR text, never image bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: int
    block_id: str
    reading_order: int
    source_coordinates: SourceCoordinates
    image_sha256: str
    ocr_status: Literal["TEXT_FOUND", "NO_TEXT", "FAILED"]
    ocr_text_sha256: str | None = None
    ocr_text_length: int = 0
    decision: AdmissionDecision | None = None
    flags: tuple[ScreeningFlag, ...] = ()


async def screen_image_crops(
    crops: tuple[UnrecognizedImageCrop, ...],
    *,
    ocr: ImageOcr,
    screening_service: ScreeningService,
) -> tuple[ImageScreeningResult, ...]:
    """OCR once as a batch, then run each nonblank OCR result through G1/G2."""
    if not crops:
        return ()
    try:
        texts = await ocr.ocr_image_crops(crops)
    except Exception:
        texts = {}
        failed = True
    else:
        failed = False

    results = []
    for crop in crops:
        image_bytes = base64.b64decode(crop.data_base64, validate=True)
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        if failed or f"{crop.page_number}:{crop.block_id}" not in texts:
            results.append(
                ImageScreeningResult(
                    page_number=crop.page_number,
                    block_id=crop.block_id,
                    reading_order=crop.reading_order,
                    source_coordinates=crop.source_coordinates,
                    image_sha256=image_digest,
                    ocr_status="FAILED",
                    decision=AdmissionDecision.QUARANTINED,
                    flags=(
                        ScreeningFlag(
                            code="IMAGE_OCR_FAILED",
                            stage="G1_INTEGRITY",
                            severity="HIGH",
                            detection_view="RAW",
                            matched_value="OCR did not return a result",
                        ),
                    ),
                )
            )
            continue
        text = texts[f"{crop.page_number}:{crop.block_id}"]
        if not text.strip():
            results.append(
                ImageScreeningResult(
                    page_number=crop.page_number,
                    block_id=crop.block_id,
                    reading_order=crop.reading_order,
                    source_coordinates=crop.source_coordinates,
                    image_sha256=image_digest,
                    ocr_status="NO_TEXT",
                )
            )
            continue
        decision, flags = screening_service.screen_extracted_text(text)
        results.append(
            ImageScreeningResult(
                page_number=crop.page_number,
                block_id=crop.block_id,
                reading_order=crop.reading_order,
                source_coordinates=crop.source_coordinates,
                image_sha256=image_digest,
                ocr_status="TEXT_FOUND",
                ocr_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                ocr_text_length=len(text),
                decision=decision,
                flags=flags,
            )
        )
    return tuple(results)
