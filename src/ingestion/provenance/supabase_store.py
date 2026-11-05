"""Supabase/PostgREST persistence for immutable ingestion provenance records."""

from __future__ import annotations

import httpx

from src.ingestion.screening.image_screening import ImageScreeningResult
from src.security.ingestion_screening.contracts import DocumentContext, ScreenedChunkedDocument


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
    ) -> None:
        """Persist chunks and decisions in one transaction via a Supabase RPC."""
        if any(
            result.source_sha256 != context.source_sha256
            for result in screened.report.chunk_results
        ):
            raise ValueError("Screening and document contexts identify different sources")
        if screened.report.document_id != screened.document.document_id:
            raise ValueError("Screening report and document IDs differ")
        document = screened.document
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
            "image_results": [
                result.model_dump(mode="json") for result in image_results
            ],
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
            raise SupabaseProvenanceError(
                "Supabase did not persist the ingestion screening batch"
            ) from exc
