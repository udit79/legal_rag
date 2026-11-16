from __future__ import annotations
import asyncio
import base64
import binascii
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Literal
import httpx
from src.ingestion.pipeline import (
    CPCRResult,
    RevisedChunkSet,
    UnrecognizedImageCrop,
)
from src.ingestion.chunk_contracts import serialize_chunk_source
from src.ingestion.retrieval_text import RETRIEVAL_TEXT_POLICY_VERSION
from src.ingestion.screening.image_screening import ImageScreeningResult
from src.ingestion.screening.image_vlm import ImageVlmResult
from src.security.ingestion_screening.enums import AdmissionDecision
from src.security.ingestion_screening.contracts import DocumentContext, ScreenedChunkedDocument

IMAGE_CROP_BUCKET = "ingestion-image-crops"
_IMAGE_UPLOAD_ATTEMPTS = 3
_IMAGE_RECONCILIATION_BATCH_SIZE = 50
_SCREENING_RPC_ATTEMPTS = 3
_LOGGER = logging.getLogger(__name__)


class SupabaseProvenanceError(RuntimeError):
    """The configured provenance database did not durably accept a batch."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        provider_error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.provider_error_code = provider_error_code
        self.request_id = request_id


@dataclass(frozen=True)
class ProvenancePersistenceReceipt:
    ingestion_run_id: str
    status: Literal["PERSISTED"]
    persisted_image_count: int
    persisted_admitted_ocr_count: int
    persisted_visual_description_count: int = 0

class SupabaseProvenanceStore:
    def __init__(self, base_url: str, service_role_key: str, client: httpx.AsyncClient):
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("Supabase URL must use HTTP or HTTPS")
        if not service_role_key.strip():
            raise ValueError("Supabase service role key must not be blank")
        self._url = base_url.rstrip("/")
        self._key = service_role_key.strip()
        self._client = client

    async def persist_screened_document(
        self,
        screened: ScreenedChunkedDocument,
        context: DocumentContext,
        *,
        image_results: tuple[ImageScreeningResult, ...] = (),
        image_vlm_results: tuple[ImageVlmResult, ...] = (),
        image_crops: tuple[UnrecognizedImageCrop, ...] = (),
    ) -> ProvenancePersistenceReceipt:
        """Persist screening and admitted OCR atomically after storing private crops."""
        ingestion_run_id = context.ingestion_run_id
        if ingestion_run_id is None:
            raise ValueError("Screening provenance persistence requires an ingestion run ID")
        if any(
            result.source_sha256 != context.source_sha256
            for result in screened.report.chunk_results
        ):
            raise ValueError("Screening and document contexts identify different sources")
        if screened.report.document_id != screened.document.document_id:
            raise ValueError("Screening report and document IDs differ")
        vlm_keys = [(result.page_number, result.block_id) for result in image_vlm_results]
        if len(set(vlm_keys)) != len(vlm_keys):
            raise SupabaseProvenanceError("VLM results contain duplicate image identities")
        document = screened.document
        image_metadata = await self._store_image_crops(
            image_crops,
            image_results,
            ingestion_run_id=ingestion_run_id,
        )
        payload = {
            "document": {
                "source_id": context.source_id,
                "source_sha256": context.source_sha256,
                "canonical_document_digest": context.canonical_document_digest,
                "source_page_count": context.source_page_count,
            },
            "run": {
                "ingestion_run_id": ingestion_run_id,
                "chunking_policy_version": context.chunking_policy_version,
                "screening_policy_version": screened.report.screening_policy_version,
                "screening_policy_digest": screened.report.screening_policy_digest,
                "trust_tier": context.trust_tier.value,
                "document_quarantined": screened.report.quarantined_document,
                "file_screening": (
                    screened.report.file_screening.model_dump(mode="json")
                    if screened.report.file_screening is not None
                    else None
                ),
            },
            "chunks": [
                {
                    "chunk_index": chunk.index,
                    "chunk_digest": result.chunk_digest,
                    "text": chunk.text,
                    "sources": [serialize_chunk_source(source) for source in chunk.sources],
                }
                for chunk, result in zip(
                    document.chunks, screened.report.chunk_results, strict=True
                )
            ],
            "screening_results": [
                result.model_dump(mode="json")
                for result in screened.report.chunk_results
            ],
            "image_results": image_metadata,
        }
        admitted_ocr = []
        for result in image_results:
            if result.ocr_text is None:
                continue
            if (
                result.ocr_status != "TEXT_FOUND"
                or result.decision is None
                or result.decision.value not in {"ADMITTED", "ADMITTED_FLAGGED"}
                or result.ocr_text_sha256 is None
                or hashlib.sha256(result.ocr_text.encode("utf-8")).hexdigest()
                != result.ocr_text_sha256
                or len(result.ocr_text) != result.ocr_text_length
            ):
                raise SupabaseProvenanceError(
                    "Only validated admitted OCR can be included in provenance persistence"
                )
            admitted_ocr.append({
                "ingestion_run_id": ingestion_run_id,
                "page_number": result.page_number,
                "block_id": result.block_id,
                "ocr_text": result.ocr_text,
                "ocr_text_sha256": result.ocr_text_sha256,
                "ocr_text_length": result.ocr_text_length,
            })

        admitted_vlm = []
        image_result_by_key = {
            (result.page_number, result.block_id): result for result in image_results
        }
        if len(image_result_by_key) != len(image_results):
            raise SupabaseProvenanceError("Image screening contains duplicate composite identities")
        for result in image_vlm_results:
            if result.status != "ADMITTED":
                continue
            image_screening = image_result_by_key.get((result.page_number, result.block_id))
            if (result.description is None or result.description_text is None
                    or result.description_sha256 is None
                    or hashlib.sha256(result.description_text.encode("utf-8")).hexdigest()
                    != result.description_sha256
                    or result.decision not in {"ADMITTED", "ADMITTED_FLAGGED"}
                    or image_screening is None
                    or image_screening.image_sha256 != result.image_sha256
                    or image_screening.ocr_status not in {"NO_TEXT", "TEXT_FOUND"}
                    or (image_screening.ocr_status == "TEXT_FOUND"
                        and image_screening.decision not in {
                            AdmissionDecision.ADMITTED,
                            AdmissionDecision.ADMITTED_FLAGGED,
                        })
                    or (image_screening.ocr_status == "NO_TEXT"
                        and image_screening.decision is not None)):
                raise SupabaseProvenanceError(
                    "Only validated admitted visual descriptions can be persisted"
                )
            admitted_vlm.append({
                "ingestion_run_id": ingestion_run_id,
                "page_number": result.page_number,
                "block_id": result.block_id,
                "image_sha256": result.image_sha256,
                "description": result.description.model_dump(mode="json"),
                "description_text": result.description_text,
                "description_sha256": result.description_sha256,
                "decision": result.decision,
                "provider": result.provider,
                "model": result.model,
                "prompt_version": result.prompt_version,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            })

        headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        request_body = {
            "p_payload": payload,
            "p_admitted_ocr": admitted_ocr,
        }
        rpc_name = "persist_ingestion_screening_with_admitted_ocr"
        if admitted_vlm:
            request_body["p_admitted_vlm"] = admitted_vlm
            rpc_name = "persist_ingestion_screening_with_admitted_ocr_and_vlm"
        endpoint = f"{self._url}/rest/v1/rpc/{rpc_name}"
        last_error: httpx.HTTPError | None = None
        for attempt in range(_SCREENING_RPC_ATTEMPTS):
            try:
                response = await self._client.post(
                    endpoint, headers=headers, json=request_body
                )
                response.raise_for_status()
                return ProvenancePersistenceReceipt(
                    ingestion_run_id=ingestion_run_id,
                    status="PERSISTED",
                    persisted_image_count=len(image_metadata),
                    persisted_admitted_ocr_count=len(admitted_ocr),
                    persisted_visual_description_count=len(admitted_vlm),
                )
            except httpx.HTTPStatusError as exc:
                last_error = exc
                retryable = (
                    exc.response.status_code in {408, 429}
                    or exc.response.status_code >= 500
                )
                if retryable and attempt + 1 < _SCREENING_RPC_ATTEMPTS:
                    await asyncio.sleep(0.05 * (2**attempt))
                    continue
                break
            except httpx.TransportError as exc:
                last_error = exc
                if attempt + 1 < _SCREENING_RPC_ATTEMPTS:
                    await asyncio.sleep(0.05 * (2**attempt))
                    continue
                break

        if last_error is None:
            raise AssertionError("Screening RPC retry loop exited without a result")
        failure_http_status = None
        failure_error_code = None
        failure_request_id = None
        failure_detail = ""
        if isinstance(last_error, httpx.HTTPStatusError):
            try:
                error_body = last_error.response.json()
            except ValueError:
                error_body = {}
            failure_error_code = (
                error_body.get("code")
                if isinstance(error_body, dict)
                and isinstance(error_body.get("code"), str)
                else None
            )
            failure_http_status = last_error.response.status_code
            failure_request_id = (
                last_error.response.headers.get("x-request-id")
                or last_error.response.headers.get("sb-request-id")
            )
            failure_message = error_body.get("message") if isinstance(error_body, dict) else None
            if not isinstance(failure_message, str):
                failure_message = ""
            failure_message = " ".join(failure_message.split())[:300]
            failure_detail = (
                f" (HTTP {failure_http_status}, Supabase code "
                f"{failure_error_code or 'unavailable'}, request ID "
                f"{failure_request_id or 'unavailable'}"
                f"{', message: ' + failure_message if failure_message else ''})"
            )
        _LOGGER.error(
            "Supabase did not atomically persist screening/OCR for run %s%s: %s",
            ingestion_run_id,
            failure_detail,
            last_error,
        )
        await self._reconcile_image_objects(
            [
                path
                for item in image_metadata
                if isinstance((path := item.get("image_object_path")), str)
            ]
        )
        raise SupabaseProvenanceError(
            "Supabase did not atomically persist the ingestion screening and OCR batch"
            f"{failure_detail}",
            http_status=failure_http_status,
            provider_error_code=failure_error_code,
            request_id=failure_request_id,
        ) from last_error

    async def _store_image_crops(
        self,
        crops: tuple[UnrecognizedImageCrop, ...],
        results: tuple[ImageScreeningResult, ...],
        *,
        ingestion_run_id: str | None,
    ) -> list[dict[str, object]]:
        crop_by_key = {(crop.page_number, crop.block_id): crop for crop in crops}
        result_by_key = {
            (result.page_number, result.block_id): result for result in results
        }
        if (
            len(crop_by_key) != len(crops)
            or len(result_by_key) != len(results)
            or crop_by_key.keys() != result_by_key.keys()
        ):
            raise SupabaseProvenanceError(
                "Image crops and screening results are not aligned"
            )
        if not crops:
            return []
        if not ingestion_run_id:
            raise SupabaseProvenanceError(
                "Image storage requires an ingestion run ID"
            )

        run_path = hashlib.sha256(ingestion_run_id.encode("utf-8")).hexdigest()
        verified_images: list[
            tuple[ImageScreeningResult, bytes, str]
        ] = []
        for key, crop in crop_by_key.items():
            result = result_by_key[key]
            try:
                image_bytes = base64.b64decode(crop.data_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise SupabaseProvenanceError(
                    "Screened image crop is not valid Base64"
                ) from exc
            image_digest = hashlib.sha256(image_bytes).hexdigest()
            if (
                not image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
                or image_digest != result.image_sha256
            ):
                raise SupabaseProvenanceError(
                    "Screened image crop failed PNG or SHA-256 verification"
                )
            verified_images.append((result, image_bytes, image_digest))

        metadata: list[dict[str, object]] = []
        uploaded_paths: set[str] = set()
        attempted_paths: list[str] = []
        for result, image_bytes, image_digest in verified_images:
            object_path = f"{run_path}/{image_digest}.png"
            if object_path not in uploaded_paths:
                attempted_paths.append(object_path)
                try:
                    await self._upload_image_crop(object_path, image_bytes)
                except SupabaseProvenanceError:
                    await self._reconcile_image_objects(attempted_paths)
                    raise
                uploaded_paths.add(object_path)
            metadata.append(
                {
                    **result.model_dump(mode="json", exclude={"ocr_text"}),
                    "image_bucket": IMAGE_CROP_BUCKET,
                    "image_object_path": object_path,
                    "image_file_size_bytes": len(image_bytes),
                }
            )
        return metadata

    async def _upload_image_crop(self, object_path: str, image_bytes: bytes) -> None:
        url = (
            f"{self._url}/storage/v1/object/"
            f"{IMAGE_CROP_BUCKET}/{object_path}"
        )
        for attempt in range(_IMAGE_UPLOAD_ATTEMPTS):
            try:
                response = await self._client.post(
                    url,
                    headers={
                        "apikey": self._key,
                        "Authorization": f"Bearer {self._key}",
                        "Content-Type": "image/png",
                        "x-upsert": "true",
                    },
                    content=image_bytes,
                )
                response.raise_for_status()
                return
            except httpx.HTTPStatusError as exc:
                if (
                    exc.response.status_code not in {408, 429}
                    and exc.response.status_code < 500
                ) or attempt + 1 == _IMAGE_UPLOAD_ATTEMPTS:
                    raise SupabaseProvenanceError(
                        "Supabase did not store an ingestion image crop"
                    ) from exc
            except httpx.TransportError as exc:
                if attempt + 1 == _IMAGE_UPLOAD_ATTEMPTS:
                    raise SupabaseProvenanceError(
                        "Supabase image storage request failed"
                    ) from exc
            await asyncio.sleep(0.05 * (2**attempt))

    async def _reconcile_image_objects(
        self, image_object_paths: list[str]
    ) -> None:
        object_paths = sorted(set(image_object_paths))
        if not object_paths:
            return

        headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
        }
        try:
            referenced_paths: set[str] = set()
            for start in range(0, len(object_paths), _IMAGE_RECONCILIATION_BATCH_SIZE):
                paths = object_paths[
                    start : start + _IMAGE_RECONCILIATION_BATCH_SIZE
                ]
                response = await self._client.get(
                    f"{self._url}/rest/v1/image_screening_results",
                    headers=headers,
                    params={
                        "select": "image_object_path",
                        "image_bucket": f"eq.{IMAGE_CROP_BUCKET}",
                        "image_object_path": f"in.({','.join(paths)})",
                    },
                )
                response.raise_for_status()
                rows = response.json()
                if not isinstance(rows, list) or any(
                    not isinstance(row, dict)
                    or not isinstance(row.get("image_object_path"), str)
                    for row in rows
                ):
                    raise SupabaseProvenanceError(
                        "Supabase returned invalid image reference data"
                    )
                referenced_paths.update(
                    row["image_object_path"] for row in rows
                )

            orphan_paths = [
                path for path in object_paths if path not in referenced_paths
            ]
            for start in range(
                0, len(orphan_paths), _IMAGE_RECONCILIATION_BATCH_SIZE
            ):
                paths = orphan_paths[
                    start : start + _IMAGE_RECONCILIATION_BATCH_SIZE
                ]
                response = await self._client.request(
                    "DELETE",
                    f"{self._url}/storage/v1/object/{IMAGE_CROP_BUCKET}",
                    headers=headers,
                    json={"prefixes": paths},
                )
                response.raise_for_status()
        except (httpx.HTTPError, SupabaseProvenanceError, ValueError):
            _LOGGER.warning(
                "Could not safely reconcile unreferenced ingestion images",
                exc_info=True,
            )

    async def persist_cpcr_output(
        self,
        revised: RevisedChunkSet,
        cpcr: CPCRResult,
        *,
        ingestion_run_id: str,
        source_sha256: str,
    ) -> None:
        """Persist the returned chunks, CPCR decisions and admission mappings."""
        if not ingestion_run_id.strip():
            raise ValueError("Ingestion run ID must not be blank")
        payload = {
            "ingestion_run_id": ingestion_run_id,
            "source_sha256": source_sha256,
            "document_id": revised.document_id,
            "source_page_count": revised.source_page_count,
            "chunking_policy_version": revised.policy_version,
            "cpcr_outcome": cpcr.outcome,
            "cpcr_candidates": [
                result.model_dump(mode="json") for result in cpcr.candidates
            ],
            "context_policy": revised.context_policy,
            "retrieval_text_policy_version": RETRIEVAL_TEXT_POLICY_VERSION,
            "contextualized_candidate_ids": list(
                revised.contextualized_candidate_ids
            ),
            "revised_chunks": [
                {
                    "index": chunk.index,
                    "text": chunk.text,
                    "retrieval_text": (
                        chunk.retrieval_text
                        if chunk.retrieval_text is not None
                        else chunk.text
                    ),
                    "retrieval_text_edits": [
                        edit.model_dump(mode="json")
                        for edit in chunk.retrieval_text_edits
                    ],
                    "chunk_sha256": hashlib.sha256(
                        chunk.text.encode("utf-8")
                    ).hexdigest(),
                    "sources": [serialize_chunk_source(source) for source in chunk.sources],
                }
                for chunk in revised.chunks
            ],
            "admission_mappings": [
                mapping.model_dump(mode="json")
                for mapping in revised.admission_mappings
            ],
            "admission_records": [
                record.model_dump(mode="json")
                for record in revised.admission_records
            ],
        }
        canonical_payload = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_digest = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
        try:
            response = await self._client.post(
                f"{self._url}/rest/v1/rpc/persist_ingestion_cpcr_output",
                headers={
                    "apikey": self._key,
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
                json={
                    "p_ingestion_run_id": ingestion_run_id,
                    "p_source_sha256": source_sha256,
                    "p_payload_digest": payload_digest,
                    "p_payload": payload,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SupabaseProvenanceError(
                "Supabase did not persist the final CPCR output"
            ) from exc
