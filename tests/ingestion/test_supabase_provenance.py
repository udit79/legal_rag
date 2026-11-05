import asyncio
import json

import httpx

from src.ingestion.chunk_contracts import Chunk, ChunkSource
from src.ingestion.provenance.supabase_store import SupabaseProvenanceStore
from src.security.ingestion_screening.contracts import DocumentContext
from src.security.ingestion_screening.enums import TrustTier
from src.security.ingestion_screening.screening_service import ScreeningService


async def _screened():
    text = "Traceable legal passage."
    chunk = Chunk(
        index=0,
        text=text,
        sources=(ChunkSource(
            page_number=1,
            block_id="page-1-block-0",
            block_reading_order=0,
            block_type="TEXT",
            source_text_length=len(text),
            start_offset=0,
            end_offset=len(text),
        ),),
    )
    context = DocumentContext(
        source_id="source-id",
        ingestion_run_id="run-id",
        source_sha256="a" * 64,
        canonical_document_digest="b" * 64,
        chunking_policy_version="5c-1.3.0+0123456789ab",
        source_page_count=1,
        trust_tier=TrustTier.T3_UNAUTHENTICATED_EXTERNAL,
    )
    screened = await ScreeningService().screen_batch(
        (chunk,), context, document_id="document-id"
    )
    return screened, context


def test_supabase_store_sends_one_rpc_with_hashes_sources_and_screening_results():
    calls = []

    async def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(204)

    async def run():
        screened, context = await _screened()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            store = SupabaseProvenanceStore(
                "https://example.supabase.co", "service-secret", client
            )
            await store.persist_screened_document(screened, context)

    asyncio.run(run())
    assert len(calls) == 1
    assert calls[0].url.path == "/rest/v1/rpc/persist_ingestion_screening"
    assert calls[0].headers["apikey"] == "service-secret"
    payload = json.loads(calls[0].content)["p_payload"]
    assert payload["document"]["source_sha256"] == "a" * 64
    assert payload["chunks"][0]["chunk_digest"]
    assert payload["screening_results"][0]["decision"] == "ADMITTED"


def test_supabase_store_reports_database_failure():
    async def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async def run():
        screened, context = await _screened()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            store = SupabaseProvenanceStore("https://example.supabase.co", "key", client)
            await store.persist_screened_document(screened, context)

    from src.ingestion.provenance.supabase_store import SupabaseProvenanceError
    import pytest

    with pytest.raises(SupabaseProvenanceError):
        asyncio.run(run())
