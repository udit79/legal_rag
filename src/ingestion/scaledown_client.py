import base64
import json
import os
from collections.abc import Mapping
from typing import Any
import httpx
from pydantic import BaseModel, ConfigDict, ValidationError
from src.ingestion.continuity_cdm_contracts import CDMClassificationRequest,CDMClassificationResult,adapt_scaledown_response

DEFAULT_SCALEDOWN_API_BASE = "https://api.scaledown.xyz"
SCALEDOWN_DECISIONS_PATH = "/v1/scaledown"
SCALEDOWN_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
SCALEDOWN_EXTRACT_MODEL = "extract-1"
_CONTINUITY_QUESTION: Mapping[str, Any] = {
    "type": "choice",
    "instructions": (
        "Determine the logical relationship between these two chunks from "
        "adjacent physical pages. Use UNKNOWN rather than guessing when the "
        "excerpts do not provide direct evidence of continuation or independence; "
        "page adjacency alone is not evidence of continuity."
    ),
    "criteria": {
        "SAME_PARAGRAPH": (
            "The target continues the same logical paragraph or sentence "
            "from the source."
        ),
        "TABLE_CONTINUATION": (
            "Both chunks belong to the same table continuing across the "
            "page boundary."
        ),
        "NO_CONTINUATION": "The chunks represent independent content.",
        "UNKNOWN": (
            "The excerpts provide insufficient or contradictory direct evidence "
            "to determine the relationship. Prefer this label over guessing."
        ),
    },
}


class ScaleDownError(Exception):
    """Base class for ScaleDown client failures."""


class ScaleDownAuthenticationError(ScaleDownError):
    """Missing or rejected API credentials."""


class ScaleDownRateLimitError(ScaleDownError):
    """The API rejected the request due to rate limiting."""


class ScaleDownQuotaError(ScaleDownError):
    """The account has insufficient quota or credit."""


class ScaleDownTimeoutError(ScaleDownError):
    """The request exceeded its configured timeout."""


class ScaleDownServerError(ScaleDownError):
    """The ScaleDown service returned a server error."""


class ScaleDownResponseError(ScaleDownError):
    """The service returned an invalid status or response payload."""


class ScaleDownExtractResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ocr_text: str
    model_id: str
    request_id: str | None = None


class ScaleDownClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base
        self._timeout = timeout
        self._http_client = http_client

    async def classify(
        self,
        request: CDMClassificationRequest,
    ) -> CDMClassificationResult:
        api_key = self._resolve_api_key()
        api_base = (
            self._api_base
            or os.environ.get("SCALEDOWN_BASE_URL")
            or DEFAULT_SCALEDOWN_API_BASE
        ).rstrip("/")
        payload = {
            "model": request.model_id,
            "state": (
                f"SOURCE PAGE {request.source.page_number}:\n"
                f"{request.source_text}\n\n"
                f"TARGET PAGE {request.target.page_number}:\n"
                f"{request.target_text}"
            ),
            "questions": {"continuity_relation": dict(_CONTINUITY_QUESTION)},
        }

        try:
            response = await self._http_client.post(
                f"{api_base}{SCALEDOWN_DECISIONS_PATH}",
                headers={"x-api-key": api_key},
                json=payload,
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise ScaleDownTimeoutError("ScaleDown request timed out") from exc
        except httpx.HTTPError as exc:
            raise ScaleDownError("ScaleDown transport request failed") from exc

        self._raise_for_status(response)
        try:
            response_body = response.json()
        except ValueError as exc:
            raise ScaleDownResponseError("ScaleDown response was not valid JSON") from exc
        try:
            result = adapt_scaledown_response(response_body, request)
            if result.request_id is None:
                request_id = (
                    response.headers.get("x-amzn-requestid")
                    or response.headers.get("x-request-id")
                    or response.headers.get("request-id")
                )
                if request_id:
                    result = result.model_copy(update={"request_id": request_id})
            return result
        except (TypeError, ValueError, ValidationError) as exc:
            raise ScaleDownResponseError(
                "ScaleDown response did not match the expected classification shape"
            ) from exc

    async def extract_document_text(
        self,
        document: bytes,
        *,
        document_mime_type: str = "application/pdf",
    ) -> ScaleDownExtractResult:
        """Ask Extract to OCR a document and return its raw recognized text."""
        if not document:
            raise ValueError("ScaleDown Extract document cannot be empty")
        if document_mime_type not in {"application/pdf", "image/png"}:
            raise ValueError("Unsupported ScaleDown Extract document MIME type")

        api_key = self._resolve_api_key()
        api_base = (
            self._api_base
            or os.environ.get("SCALEDOWN_BASE_URL")
            or DEFAULT_SCALEDOWN_API_BASE
        ).rstrip("/")
        encoded_document = base64.b64encode(document).decode("ascii")
        payload = {
            "model": SCALEDOWN_EXTRACT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Read the visible text from this single PDF page.",
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "page_text",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "visible_text": {
                                "type": "string",
                                "description": "Visible text on this page",
                            }
                        },
                    },
                },
            },
            "document": encoded_document,
            "document_mime_type": document_mime_type,
        }
        try:
            response = await self._http_client.post(
                f"{api_base}{SCALEDOWN_CHAT_COMPLETIONS_PATH}",
                headers={"x-api-key": api_key},
                json=payload,
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise ScaleDownTimeoutError("ScaleDown Extract request timed out") from exc
        except httpx.HTTPError as exc:
            raise ScaleDownError("ScaleDown Extract transport request failed") from exc

        self._raise_for_status(response)
        try:
            response_body = response.json()
            model_id = response_body["model"]
            message_content = response_body["choices"][0]["message"]["content"]
            extracted = json.loads(message_content)
            ocr_text = extracted["ocr_text"]
            if model_id != SCALEDOWN_EXTRACT_MODEL:
                raise ValueError("ScaleDown returned an unexpected Extract model")
            if not isinstance(ocr_text, str):
                raise ValueError("ScaleDown Extract returned a non-string ocr_text")
            request_id = response_body.get("id") or response.headers.get(
                "x-amzn-requestid"
            ) or response.headers.get("x-request-id")
            return ScaleDownExtractResult(
                ocr_text=ocr_text,
                model_id=model_id,
                request_id=request_id,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise ScaleDownResponseError(
                "ScaleDown Extract response did not contain valid OCR text"
            ) from exc

    def _resolve_api_key(self) -> str:
        api_key = self._api_key or os.environ.get("SCALEDOWN_API_KEY")
        if not api_key or not api_key.strip():
            raise ScaleDownAuthenticationError(
                "SCALEDOWN_API_KEY is not configured"
            )
        return api_key.strip()

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status_code = response.status_code
        if status_code in {401, 403}:
            raise ScaleDownAuthenticationError(
                f"ScaleDown rejected authentication (HTTP {status_code})"
            )
        if status_code == 402:
            raise ScaleDownQuotaError("ScaleDown quota or credit is unavailable")
        if status_code == 429:
            raise ScaleDownRateLimitError("ScaleDown rate limit exceeded")
        if status_code >= 500:
            raise ScaleDownServerError(
                f"ScaleDown server error (HTTP {status_code})"
            )
        if status_code >= 400:
            raise ScaleDownResponseError(
                f"ScaleDown rejected the request (HTTP {status_code})"
            )
        if not 200 <= status_code < 300:
            raise ScaleDownResponseError(
                f"Unexpected ScaleDown HTTP status: {status_code}"
            )
