from typing import Annotated, Literal, Self
from pydantic import BaseModel, Field, model_validator,StrictFloat,StringConstraints

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

class DocumentRef(BaseModel):
    document_id: Annotated[str,Field(min_length=1)]
    source_ref: Annotated[str,Field(min_length=1)]
    filename: Annotated[str,Field(min_length=1)]
    media_type: DocumentMediaType

NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
class DocumentFeatures(BaseModel):
    content_sha256: Annotated[str,Field(pattern=r"^[0-9a-fA-F]{64}$")]
    file_size_bytes: NonNegativeInt
    page_count: NonNegativeInt
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
    route: Route
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    decision_source: DecisionSource
    confidence: Annotated[StrictFloat|None,Field(ge=0.0, le=1.0)]=None
    policy_version: str = Field(min_length=1)
