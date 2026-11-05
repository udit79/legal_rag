"""Deterministic, behavior-versioned screening policy."""

import hashlib
import json
import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from src.ingestion.contracts import NonEmptyStr

_Pattern = Annotated[str, StringConstraints(min_length=1)]


class ScreeningPolicy(BaseModel):
    """All configurable thresholds and patterns that affect screening output."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    version: NonEmptyStr = "isg-1.2.0"
    boundary_window_chars: Annotated[int, Field(ge=1, strict=True)] = 300
    max_file_bytes: Annotated[int, Field(ge=1, strict=True)] = 50 * 1024 * 1024
    max_pdf_pages: Annotated[int, Field(ge=1, strict=True)] = 1000
    max_pdf_objects: Annotated[int, Field(ge=1, strict=True)] = 200_000
    max_nonprinting_ratio: Annotated[
        float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)
    ] = 0.02
    max_bidi_controls: Annotated[int, Field(ge=0, strict=True)] = 8
    max_zero_width_run: Annotated[int, Field(ge=0, strict=True)] = 8
    min_encoded_blob_chars: Annotated[int, Field(ge=1, strict=True)] = 256
    min_mixed_script_token_chars: Annotated[int, Field(ge=2, strict=True)] = 4
    max_flagged_chunk_ratio: Annotated[
        float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)
    ] | None = None
    decoded_tag_instruction_patterns: tuple[_Pattern, ...] = (
        r"(?is)\b(?:ignore|disregard|override)\s+(?:(?:all|the|previous|system|developer|above)\s+){0,3}instructions\b",
        r"(?is)\b(?:reveal|print|show|expose)\s+(?:the\s+)?(?:system|developer)\s+prompt\b",
    )
    model_directed_patterns: tuple[_Pattern, ...] = (
        r"(?is)(?:AI[- ]assistant|language\s+model|ChatGPT|LLM|system\s+prompt|developer\s+prompt).{0,80}(?:ignore|disregard|override|reveal|exfiltrate).{0,80}(?:instructions|prompt|data|tool)",
        r"(?is)(?:reveal|print|show)\s+(?:the\s+)?(?:system|developer)\s+prompt",
        r"(?is)(?:AI[- ]assistant|language\s+model|ChatGPT|LLM).{0,80}(?:execute|run)\s+(?:this|the\s+following)\s+(?:tool|command)",
    )
    high_severity_patterns: tuple[_Pattern, ...] = (
        r"(?is)\b(?:AI[- ]assistant|language\s+model|ChatGPT|LLM)\b.{0,80}\b(?:ignore|override|reveal|exfiltrate)\b",
        r"(?is)<\|im_start\|>\s*(?:system|developer)\b.{0,300}<\|im_end\|>",
        r"(?is)<\|assistant\b[^>]{0,120}\bto\s*=\s*(?:python|browser|[\w.-]+)\b",
        r"(?is)<\|(?:tool_call|function_call)\|>",
        r"(?is)<!--.{0,200}(?:ignore|override|system\s+prompt|exfiltrat).{0,200}-->",
        r"(?is)<\s*script\b[^>]*>",
        r"(?i)\bjavascript\s*:",
        r"(?is)\bon\s*(?:error|load|click|mouseover)\s*=\s*[\"'][^\"']{1,300}[\"']",
        r"(?i)!\[[^\]]*\]\(\s*https?://[^)]+\)",
        r"(?is)<\s*(?:iframe|object|embed)\b[^>]*>",
        r"(?is)<\s*img\b[^>]*\bsrc\s*=\s*[\"']?https?://[^\s\"'>]+",
        r"(?is)\bdata\s*:\s*(?:text/html|application/(?:javascript|xhtml\+xml))",
    )
    low_severity_patterns: tuple[_Pattern, ...] = (
        r"(?i)<\|\s*(?:system|assistant|developer|user|tool)\s*\|>",
    )

    @field_validator(
        "decoded_tag_instruction_patterns",
        "model_directed_patterns",
        "high_severity_patterns",
        "low_severity_patterns",
    )
    @classmethod
    def validate_patterns(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid screening regex: {pattern!r}") from exc
        return patterns

    def digest(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
