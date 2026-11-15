from typing import Annotated, Literal, Self
import re
import hashlib
from pydantic import BaseModel,ConfigDict,Field,StringConstraints,model_validator
from src.ingestion.contracts import DocumentDisposition, PositiveInt
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
BlockType = Literal["TEXT", "TABLE", "IMAGE", "VECTOR", "OTHER"]
ExtractionMethod = Literal["NATIVE", "IMAGE_OCR", "IMAGE_VLM"]
PageParseStatus = Literal["PARSED", "BLANK", "VECTOR_ONLY", "FAILED"]

class SourceCoordinates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    x0: float = Field(allow_inf_nan=False)
    y0: float = Field(allow_inf_nan=False)
    x1: float = Field(allow_inf_nan=False)
    y1: float = Field(allow_inf_nan=False)
    unit: Literal["PDF_POINT", "PIXEL"]
    origin: Literal["TOP_LEFT", "BOTTOM_LEFT"]
    @model_validator(mode="after")
    def validate_bounds(self) -> "SourceCoordinates":
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Source coordinate bounds must be ordered")
        return self

class CanonicalTextTransformation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    original_character: Annotated[str, Field(min_length=1, max_length=1)]
    replacement: Annotated[str, Field(min_length=1, max_length=1)]
    source_page: PositiveInt
    start_offset: NonNegativeInt
    end_offset: PositiveInt
    reason: NonEmptyStr

    @model_validator(mode="after")
    def validate_single_character_range(self) -> Self:
        if self.end_offset != self.start_offset + 1:
            raise ValueError("Character transformations must cover one source character")
        return self

class CanonicalBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: PositiveInt
    text: str
    block_type: BlockType
    reading_order: NonNegativeInt
    block_id: NonEmptyStr = Field(default="", validate_default=True)
    source_coordinates: SourceCoordinates
    confidence: Annotated[float, Field(ge=0.0, le=100.0)] | None = None
    raw_text: str | None = None
    text_transformations: tuple[CanonicalTextTransformation, ...] = ()
    extraction_method: ExtractionMethod = "NATIVE"
    source_image_block_id: str | None = None
    ocr_text_sha256: str | None = None
    visual_description_sha256: str | None = None
    visual_model_provider: str | None = None
    visual_model: str | None = None
    visual_prompt_version: str | None = None
    derived_order: NonNegativeInt | None = None

    @model_validator(mode="before")
    @classmethod
    def assign_default_block_id(cls, value: object) -> object:
        if isinstance(value, dict) and "block_id" not in value:
            value = dict(value)
            value["block_id"] = canonical_block_id(
                value["page_number"], value["reading_order"]
            )
        return value

    @model_validator(mode="after")
    def validate_block_id_scope(self) -> "CanonicalBlock":
        prefix = f"page-{self.page_number}-block-"
        suffix = self.block_id.removeprefix(prefix)
        native_id = suffix.isascii() and suffix.isdecimal()
        derived_id = re.fullmatch(r"(?:ocr|vlm)-[0-9a-f]{64}", suffix) is not None
        if not self.block_id.startswith(prefix) or not (native_id or derived_id):
            raise ValueError("Block ID must retain its page-scoped canonical format")
        if self.extraction_method == "IMAGE_OCR":
            if (self.block_type != "IMAGE" or not self.source_image_block_id
                    or self.ocr_text_sha256 is None
                    or self.derived_order is None
                    or self.derived_order == 0
                    or not re.fullmatch(r"[0-9a-f]{64}", self.ocr_text_sha256)):
                raise ValueError("Image OCR blocks require source image identity and text digest")
            if self.ocr_text_sha256 != hashlib.sha256(self.text.encode("utf-8")).hexdigest():
                raise ValueError("Image OCR digest does not match canonical text")
            if any((self.visual_description_sha256, self.visual_model_provider, self.visual_model, self.visual_prompt_version)):
                raise ValueError("OCR blocks cannot carry VLM provenance")
        elif self.extraction_method == "IMAGE_VLM":
            if (self.block_type != "IMAGE" or not self.source_image_block_id
                    or self.derived_order is None or self.derived_order < 2
                    or self.ocr_text_sha256 is not None
                    or not self.visual_description_sha256
                    or not re.fullmatch(r"[0-9a-f]{64}", self.visual_description_sha256)
                    or self.visual_description_sha256 != hashlib.sha256(self.text.encode("utf-8")).hexdigest()
                    or not self.visual_model_provider or not self.visual_model
                    or not self.visual_prompt_version
                    or not self.visual_prompt_version.strip()
                    or self.block_id != canonical_visual_block_id(
                        self.page_number,
                        self.source_image_block_id,
                        self.visual_model_provider,
                        self.visual_model,
                        self.visual_prompt_version,
                        self.visual_description_sha256,
                    )):
                raise ValueError("VLM blocks require image identity, digest, model provenance and ordering")
        elif (
            self.source_image_block_id is not None
            or self.ocr_text_sha256 is not None
            or self.derived_order is not None
            or self.visual_description_sha256 is not None
            or self.visual_model_provider is not None
            or self.visual_model is not None
            or self.visual_prompt_version is not None
        ):
            raise ValueError("Native blocks cannot carry derived image provenance")
        if self.text_transformations:
            if self.raw_text is None or len(self.raw_text) != len(self.text):
                raise ValueError("Transformed blocks require same-length raw source text")
            for edit in self.text_transformations:
                if (
                    edit.source_page != self.page_number
                    or edit.end_offset > len(self.raw_text)
                    or self.raw_text[edit.start_offset : edit.end_offset]
                    != edit.original_character
                    or self.text[edit.start_offset : edit.end_offset]
                    != edit.replacement
                ):
                    raise ValueError("Text transformation does not match source offsets")
        return self

def canonical_block_id(page_number: int, reading_order: int) -> str:return f"page-{page_number}-block-{reading_order}"


def canonical_visual_block_id(
    page_number: int,
    source_image_block_id: str,
    provider: str,
    model: str,
    prompt_version: str,
    description_sha256: str,
) -> str:
    identity = hashlib.sha256(
        "\0".join((
            str(page_number), source_image_block_id, provider, model,
            prompt_version, description_sha256,
        )).encode("utf-8")
    ).hexdigest()
    return f"page-{page_number}-block-vlm-{identity}"

class ParserFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parser: NonEmptyStr
    code: NonEmptyStr
    message: NonEmptyStr

class CanonicalPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    page_number: PositiveInt
    parser: NonEmptyStr
    status: PageParseStatus
    blocks: tuple[CanonicalBlock, ...] = ()
    failure: ParserFailure | None = None
    warnings: tuple[ParserFailure, ...] = ()

    @model_validator(mode="after")
    def validate_page_result(self) -> "CanonicalPage":
        if any(block.page_number != self.page_number for block in self.blocks):
            raise ValueError("Every block must belong to its containing page")
        original_reading_orders = tuple(
            block.reading_order
            for block in self.blocks
            if block.extraction_method == "NATIVE"
        )
        if original_reading_orders != tuple(range(len(original_reading_orders))):
            raise ValueError("Original parser block reading order must remain contiguous")
        ordering_keys = tuple(
            (block.reading_order, block.derived_order or 0)
            for block in self.blocks
        )
        if ordering_keys != tuple(sorted(ordering_keys)) or len(ordering_keys) != len(set(ordering_keys)):
            raise ValueError("Canonical block order must be stable and unique")
        block_ids = tuple(block.block_id for block in self.blocks)
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("Block IDs must be unique within a page")
        if self.status == "FAILED":
            if self.failure is None or self.blocks:
                raise ValueError(
                    "Failed pages require a failure and cannot have blocks"
                )
        elif self.failure is not None:
            raise ValueError("Only failed pages may contain a failure")
        if self.status == "BLANK" and self.blocks:
            raise ValueError("Blank pages cannot contain blocks")
        if self.status == "VECTOR_ONLY" and (
            not self.blocks
            or any(block.block_type != "VECTOR" for block in self.blocks)
        ):
            raise ValueError("Vector-only pages must contain only vector blocks")
        if self.status == "PARSED" and not self.blocks:
            raise ValueError("Parsed pages must contain at least one block")
        return self

class CanonicalDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    page_count: PositiveInt
    document_disposition: DocumentDisposition
    pages: tuple[CanonicalPage, ...]
    document_failure: ParserFailure | None = None

    @model_validator(mode="after")
    def validate_pages(self) -> "CanonicalDocument":
        expected_page_numbers = tuple(range(1, self.page_count + 1))
        actual_page_numbers = tuple(page.page_number for page in self.pages)
        if actual_page_numbers != expected_page_numbers:
            raise ValueError(
                "Canonical pages must cover every source page exactly once "
                "in ascending order"
            )
        if self.document_failure is not None and any(
            page.status != "FAILED" for page in self.pages
        ):
            raise ValueError(
                "A document-level failure requires every page to be marked failed"
            )
        if self.document_disposition == "REJECT_NO_CONTENT" and any(
            page.status in {"PARSED", "VECTOR_ONLY"} for page in self.pages
        ):
            raise ValueError(
                "A document containing parsed content cannot be rejected as empty"
            )
        return self
