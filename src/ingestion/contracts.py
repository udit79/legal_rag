from typing import Annotated, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

DocumentMediaType = Literal["application/pdf"]
Route = Literal[
    "FAST_TEXT",
    "STRUCTURED_PARSE",
    "OCR",
    "LAYOUT_OCR",
    "SKIP_BLANK",
    "ESCALATE",
]
DecisionSource = Literal["RULES", "CDM", "FALLBACK"]
DocumentDisposition = Literal["PROCESS", "REJECT_NO_CONTENT"]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

class DocumentRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: NonEmptyStr
    source_ref: NonEmptyStr
    filename: NonEmptyStr
    media_type: DocumentMediaType

class PageFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_number: PositiveInt
    text_char_count: NonNegativeInt
    image_count: NonNegativeInt
    drawing_count: NonNegativeInt


class DocumentFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_sha256: Annotated[
        str, StringConstraints(pattern=r"^[0-9a-fA-F]{64}$", to_lower=True)
    ]
    file_size_bytes: Annotated[int, Field(gt=0, strict=True)]
    page_count: PositiveInt
    pages: tuple[PageFeatures, ...]

    @model_validator(mode="after")
    def validate_pages(self) -> Self:
        if len(self.pages) != self.page_count:
            raise ValueError("Number of page records must equal page_count")
        expected_numbers = tuple(range(1, self.page_count + 1))
        actual_numbers = tuple(page.page_number for page in self.pages)
        if actual_numbers != expected_numbers:
            raise ValueError("Page numbers must be contiguous and one-based")
        return self


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route: Route
    reason: NonEmptyStr
    decision_source: DecisionSource
    confidence: Annotated[float, Field(ge=0.0, le=1.0, strict=True)] | None = None
    policy_version: NonEmptyStr

class PageRouteDecision(RouteDecision):
    page_number: PositiveInt

class DocumentRoutePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_count: PositiveInt
    decisions: tuple[PageRouteDecision, ...]
    document_disposition: DocumentDisposition

    @model_validator(mode="after")
    def validate_page_decisions(self) -> Self:
        page_numbers = tuple(decision.page_number for decision in self.decisions)
        expected_numbers = tuple(range(1, self.page_count + 1))
        if page_numbers != expected_numbers:
            raise ValueError(
                "Page route decisions must cover every page exactly once "
                "in ascending page order"
            )
        policy_versions = {decision.policy_version for decision in self.decisions}
        if len(policy_versions) != 1:
            raise ValueError("All page decisions must use the same policy version")

        all_pages_blank = all(
            decision.route == "SKIP_BLANK" for decision in self.decisions
        )
        if all_pages_blank and self.document_disposition != "REJECT_NO_CONTENT":
            raise ValueError(
                "A document with all pages marked SKIP_BLANK must be rejected "
                "for no detected content"
            )
        if not all_pages_blank and self.document_disposition == "REJECT_NO_CONTENT":
            raise ValueError(
                "A no-content disposition requires all pages to be SKIP_BLANK"
            )
        return self
