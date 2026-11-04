from typing import Annotated, Literal
from pydantic import BaseModel,ConfigDict,Field,StringConstraints,model_validator
from src.ingestion.contracts import DocumentDisposition, PositiveInt

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
BlockType = Literal["TEXT", "TABLE", "IMAGE", "VECTOR", "OTHER"]
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

class CanonicalBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: PositiveInt
    text: str
    block_type: BlockType
    reading_order: NonNegativeInt
    block_id: NonEmptyStr = Field(default="", validate_default=True)
    source_coordinates: SourceCoordinates
    confidence: Annotated[float, Field(ge=0.0, le=100.0)] | None = None

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
        if not self.block_id.startswith(prefix) or not suffix.isascii() or not suffix.isdecimal():
            raise ValueError("Block ID must retain its page-scoped canonical format")
        return self

def canonical_block_id(page_number: int, reading_order: int) -> str:return f"page-{page_number}-block-{reading_order}"

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
        reading_orders = tuple(block.reading_order for block in self.blocks)
        if reading_orders != tuple(range(len(self.blocks))):
            raise ValueError("Block reading order must be contiguous and ascending")
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
