from __future__ import annotations
import asyncio
import base64
import binascii
import hashlib
import json
import logging
import httpx
from src.ingestion.pipeline import (
    CPCRResult,
    RevisedChunkSet,
    UnrecognizedImageCrop,
)
from src.ingestion.retrieval_text import RETRIEVAL_TEXT_POLICY_VERSION
from src.ingestion.screening.image_screening import ImageScreeningResult
from src.security.ingestion_screening.contracts import DocumentContext, ScreenedChunkedDocument

IMAGE_CROP_BUCKET = "ingestion-image-crops"
_IMAGE_UPLOAD_ATTEMPTS = 3
_IMAGE_RECONCILIATION_BATCH_SIZE = 50
_LOGGER = logging.getLogger(__name__)


class SupabaseProvenanceError(RuntimeError):
    """The configured provenance database did not durably accept a batch."""

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
        image_crops: tuple[UnrecognizedImageCrop, ...] = (),
    ) -> None:
        """Store verified private crops, then persist screening metadata by RPC."""
        if any(
            result.source_sha256 != context.source_sha256
            for result in screened.report.chunk_results
        ):
            raise ValueError("Screening and document contexts identify different sources")
        if screened.report.document_id != screened.document.document_id:
            raise ValueError("Screening report and document IDs differ")
        document = screened.document
        image_metadata = await self._store_image_crops(
            image_crops,
            image_results,
            ingestion_run_id=context.ingestion_run_id,
        )
        payload = {
            "document": {
                "source_id": context.source_id,
                "source_sha256": context.source_sha256,
                "canonical_document_digest": context.canonical_document_digest,
                "source_page_count": context.source_page_count,
            },
            "run": {
                "ingestion_run_id": context.ingestion_run_id,
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
                    "sources": [source.model_dump(mode="json") for source in chunk.sources],
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
        try:
            response = await self._client.post(
                f"{self._url}/rest/v1/rpc/persist_ingestion_screening",
                headers={
                    "apikey": self._key,
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
                json={"p_payload": payload},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                await self._reconcile_image_objects(image_metadata)
            elif image_metadata:
                _LOGGER.warning(
                    "Ingestion metadata RPC outcome is unknown; retaining uploaded images"
                )
            raise SupabaseProvenanceError(
                "Supabase did not persist the ingestion screening batch"
            ) from exc

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
        for result, image_bytes, image_digest in verified_images:
            object_path = f"{run_path}/{image_digest}.png"
            if object_path not in uploaded_paths:
                await self._upload_image_crop(object_path, image_bytes)
                uploaded_paths.add(object_path)
            metadata.append(
                {
                    **result.model_dump(mode="json"),
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
        self, image_metadata: list[dict[str, object]]
    ) -> None:
        object_paths = sorted(
            {
                path
                for item in image_metadata
                if item.get("image_bucket") == IMAGE_CROP_BUCKET
                and isinstance((path := item.get("image_object_path")), str)
            }
        )
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
                    "sources": [source.model_dump(mode="json") for source in chunk.sources],
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
