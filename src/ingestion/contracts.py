from typing import Annotated, Literal, Self
from pydantic import BaseModel, Field, model_validator,StrictFloat,StringConstraints,ConfigDict

DocumentMediaType = Literal["application/pdf"]
Route = Literal[
    "FAST_TEXT",
    "STRUCTURED_PARSE",
    "OCR",
    "LAYOUT_OCR",
    "ESCALATE",
    "REJECT",
]
DecisionSource = Literal["RULES", "CDM", "FALLBACK"]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
NonEmptyStr=Annotated[str,StringConstraints(strip_whitespace=True, min_length=1)]
class DocumentRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: NonEmptyStr
    source_ref: NonEmptyStr
    filename: NonEmptyStr
    media_type: DocumentMediaType


class DocumentFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_sha256: Annotated[str,StringConstraints(pattern=r"^[0-9a-fA-F]{64}$",to_lower=True)]
    file_size_bytes: Annotated[int, Field(gt=0, strict=True)]
    page_count: Annotated[int, Field(gt=0, strict=True)]
    page_text_char_counts: list[NonNegativeInt]
    page_image_counts: list[NonNegativeInt]
    @model_validator(mode="after")
    def validate_page_counts(self) -> Self:
        if len(self.page_text_char_counts) != self.page_count:
            raise ValueError(
                "Number of pages is not equal to the character generate for each page."
            )
        if len(self.page_image_counts) != self.page_count:
            raise ValueError("Number of pages is not equal to the image list for each page.")
        return self
        


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route: Route
    reason: NonEmptyStr
    decision_source: DecisionSource
    confidence: Annotated[StrictFloat|None,Field(ge=0.0, le=1.0)]=None
    policy_version: NonEmptyStr
