"""Validated JSON messages exchanged with the parser container."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ingestion.contracts import DocumentFeatures, DocumentRoutePlan, NonEmptyStr
from src.ingestion.parser_contracts import CanonicalDocument


class ParseWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["parse"] = "parse"
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    filename: str = Field(pattern=r"^[0-9a-f]{32}\.pdf$")
    routing_policy_version: NonEmptyStr


class ParseWorkerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["ready", "result", "error"]
    request_id: str | None = None
    features: DocumentFeatures | None = None
    route_plan: DocumentRoutePlan | None = None
    canonical: CanonicalDocument | None = None
    error_code: NonEmptyStr | None = None
    error_type: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_message(self) -> Self:
        if self.type == "ready":
            if any((self.request_id, self.features, self.route_plan, self.canonical, self.error_code, self.error_type)):
                raise ValueError("Ready messages cannot contain request data")
        elif self.type == "result":
            if not all((self.request_id, self.features, self.route_plan, self.canonical)):
                raise ValueError("Result messages require all parsed document data")
            if self.error_code is not None or self.error_type is not None:
                raise ValueError("Result messages cannot contain error fields")
        elif self.type == "error":
            if not self.request_id or not self.error_code or not self.error_type:
                raise ValueError("Error messages require an ID and error details")
            if any((self.features, self.route_plan, self.canonical)):
                raise ValueError("Error messages cannot contain parsed document data")
        return self


class ImageOcrInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_id: NonEmptyStr
    data_base64: NonEmptyStr


class ImageOcrRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["ocr_images"] = "ocr_images"
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    images: tuple[ImageOcrInput, ...]


class ImageOcrText(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_id: NonEmptyStr
    text: str


class ImageOcrResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["ocr_result"]
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    results: tuple[ImageOcrText, ...]


class PagePdfRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["page_pdf"] = "page_pdf"
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    filename: str = Field(pattern=r"^[0-9a-f]{32}\.pdf$")
    page_number: int = Field(ge=1, strict=True)


class PagePdfResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["page_pdf_result"]
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    page_number: int = Field(ge=1, strict=True)
    width_points: float = Field(gt=0, allow_inf_nan=False)
    height_points: float = Field(gt=0, allow_inf_nan=False)
    data_base64: NonEmptyStr
