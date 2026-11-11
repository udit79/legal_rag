import asyncio
import base64
import hashlib
import json

import httpx

from src.ingestion.chunk_contracts import Chunk, ChunkSource
from src.ingestion.parser_contracts import SourceCoordinates
from src.ingestion.pipeline import (
    CPCRResult,
    RevisedChunkAdmissionMapping,
    RevisedChunkSet,
    UnrecognizedImageCrop,
)
from src.ingestion.provenance.supabase_store import (
    IMAGE_CROP_BUCKET,
    SupabaseProvenanceError,
    SupabaseProvenanceStore,
)
from src.ingestion.screening.image_screening import ImageScreeningResult
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


def test_supabase_store_persists_final_cpcr_chunks_and_decisions():
    calls = []

    async def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(204)

    async def run():
        screened, context = await _screened()
        mapping = RevisedChunkAdmissionMapping(
            revised_chunk_index=0,
            original_chunk_indices=(0,),
            admission_records=screened.admission_records,
        )
        revised = RevisedChunkSet(
            document_id=screened.document.document_id,
            source_page_count=1,
            policy_version=screened.document.policy_version,
            context_policy="test context overlap policy",
            chunks=screened.document.chunks,
            contextualized_candidate_ids=(),
            admission_records=screened.admission_records,
            admission_mappings=(mapping,),
            screening_report=screened.report,
        )
        cpcr = CPCRResult(outcome="COMPLETED", candidates=())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            store = SupabaseProvenanceStore("https://example.supabase.co", "key", client)
            await store.persist_cpcr_output(
                revised,
                cpcr,
                ingestion_run_id="run-id",
                source_sha256=context.source_sha256,
            )

    asyncio.run(run())
    assert len(calls) == 1
    assert calls[0].url.path == "/rest/v1/rpc/persist_ingestion_cpcr_output"
    payload = json.loads(calls[0].content)
    assert payload["p_payload_digest"]
    assert (
        payload["p_payload"]["retrieval_text_policy_version"]
        == "retrieval-text-1.0.0"
    )
    assert payload["p_payload"]["revised_chunks"][0]["retrieval_text"]
    assert payload["p_payload"]["revised_chunks"][0]["chunk_sha256"]
    assert payload["p_payload"]["admission_mappings"][0]["original_chunk_indices"] == [0]


def _image_crop(block_id: str, image_bytes: bytes) -> UnrecognizedImageCrop:
    return UnrecognizedImageCrop(
        page_number=1,
        block_id=block_id,
        reading_order=0,
        source_coordinates=SourceCoordinates(
            x0=0, y0=0, x1=10, y1=10, unit="PDF_POINT", origin="TOP_LEFT"
        ),
        filename=f"{block_id}.png",
        data_base64=base64.b64encode(image_bytes).decode("ascii"),
    )


def _image_result(crop: UnrecognizedImageCrop, image_bytes: bytes):
    return ImageScreeningResult(
        page_number=crop.page_number,
        block_id=crop.block_id,
        reading_order=crop.reading_order,
        source_coordinates=crop.source_coordinates,
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        ocr_status="NO_TEXT",
    )


def test_image_upload_retries_and_persists_private_object_metadata():
    png = b"\x89PNG\r\n\x1a\nverified-crop"
    crop = _image_crop("page-1-block-0", png)
    result = _image_result(crop, png)
    requests = []
    storage_attempts = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal storage_attempts
        requests.append(request)
        if request.url.path.startswith("/storage/v1/object/"):
            storage_attempts += 1
            if storage_attempts == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"Key": request.url.path})
        return httpx.Response(204)

    async def run():
        screened, context = await _screened()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            store = SupabaseProvenanceStore("https://example.supabase.co", "key", client)
            await store.persist_screened_document(
                screened,
                context,
                image_results=(result,),
                image_crops=(crop,),
            )
            await store.persist_screened_document(
                screened,
                context,
                image_results=(result,),
                image_crops=(crop,),
            )

    asyncio.run(run())
    storage_requests = [
        request for request in requests
        if request.url.path.startswith("/storage/v1/object/")
    ]
    rpc_requests = [
        request for request in requests
        if request.url.path.endswith("/persist_ingestion_screening")
    ]
    assert storage_attempts == 3
    assert storage_requests[0].url.path == storage_requests[1].url.path
    assert storage_requests[1].url.path == storage_requests[2].url.path
    assert storage_requests[1].headers["x-upsert"] == "true"
    assert storage_requests[1].headers["content-type"] == "image/png"
    assert storage_requests[1].content == png
    assert len(rpc_requests) == 2
    metadata = json.loads(rpc_requests[0].content)["p_payload"]["image_results"][0]
    assert metadata["image_bucket"] == IMAGE_CROP_BUCKET
    assert metadata["image_object_path"] in storage_requests[1].url.path
    assert metadata["image_sha256"] == hashlib.sha256(png).hexdigest()
    assert metadata["image_file_size_bytes"] == len(png)
    assert "data_base64" not in metadata
    assert base64.b64encode(png).decode("ascii") not in rpc_requests[0].content.decode()


def test_image_upload_rejects_hash_mismatch_before_storage_or_rpc():
    png = b"\x89PNG\r\n\x1a\nverified-crop"
    crop = _image_crop("page-1-block-0", png)
    result = _image_result(crop, png).model_copy(
        update={"image_sha256": "0" * 64}
    )
    requests = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    async def run():
        screened, context = await _screened()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            store = SupabaseProvenanceStore("https://example.supabase.co", "key", client)
            await store.persist_screened_document(
                screened,
                context,
                image_results=(result,),
                image_crops=(crop,),
            )

    import pytest

    with pytest.raises(SupabaseProvenanceError, match="SHA-256 verification"):
        asyncio.run(run())
    assert requests == []


def test_partial_image_upload_failure_does_not_persist_metadata():
    first_png = b"\x89PNG\r\n\x1a\nfirst-crop"
    second_png = b"\x89PNG\r\n\x1a\nsecond-crop"
    crops = (
        _image_crop("page-1-block-0", first_png),
        _image_crop("page-1-block-1", second_png),
    )
    results = (
        _image_result(crops[0], first_png),
        _image_result(crops[1], second_png),
    )
    requests = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if (
            request.method == "POST"
            and request.url.path.endswith(
                f"{hashlib.sha256(second_png).hexdigest()}.png"
            )
        ):
            return httpx.Response(503)
        if (
            request.method == "POST"
            and request.url.path.startswith("/storage/v1/object/")
        ):
            return httpx.Response(200)
        if request.method == "GET" and request.url.path.endswith(
            "/image_screening_results"
        ):
            return httpx.Response(200, json=[])
        if request.method == "DELETE":
            return httpx.Response(200, json={"message": "deleted"})
        return httpx.Response(204)

    async def run():
        screened, context = await _screened()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            store = SupabaseProvenanceStore("https://example.supabase.co", "key", client)
            await store.persist_screened_document(
                screened,
                context,
                image_results=results,
                image_crops=crops,
            )

    import pytest

    with pytest.raises(SupabaseProvenanceError, match="did not store"):
        asyncio.run(run())
    delete_requests = [
        request for request in requests if request.method == "DELETE"
    ]
    assert sum(
        request.url.path.endswith("/persist_ingestion_screening")
        for request in requests
    ) == 0
    assert len(delete_requests) == 1
    deleted_paths = json.loads(delete_requests[0].content)["prefixes"]
    assert deleted_paths == sorted(
        [
            f"{hashlib.sha256(b'run-id').hexdigest()}/{hashlib.sha256(first_png).hexdigest()}.png",
            f"{hashlib.sha256(b'run-id').hexdigest()}/{hashlib.sha256(second_png).hexdigest()}.png",
        ]
    )


def test_rpc_failure_only_deletes_images_confirmed_unreferenced():
    import pytest

    png = b"\x89PNG\r\n\x1a\norphan-candidate"
    crop = _image_crop("page-1-block-0", png)
    result = _image_result(crop, png)
    object_path = (
        f"{hashlib.sha256(b'run-id').hexdigest()}/"
        f"{hashlib.sha256(png).hexdigest()}.png"
    )

    async def persist_with_reference_state(referenced: bool | None):
        requests = []

        async def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "POST" and request.url.path.startswith(
                "/storage/v1/object/"
            ):
                return httpx.Response(200)
            if request.url.path.endswith("/persist_ingestion_screening"):
                return httpx.Response(500)
            if request.method == "GET" and request.url.path.endswith(
                "/image_screening_results"
            ):
                if referenced is None:
                    return httpx.Response(503)
                rows = (
                    [{"image_object_path": object_path}] if referenced else []
                )
                return httpx.Response(200, json=rows)
            if request.method == "DELETE":
                return httpx.Response(200, json={"message": "deleted"})
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        screened, context = await _screened()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle)
        ) as client:
            store = SupabaseProvenanceStore(
                "https://example.supabase.co", "key", client
            )
            with pytest.raises(SupabaseProvenanceError):
                await store.persist_screened_document(
                    screened,
                    context,
                    image_results=(result,),
                    image_crops=(crop,),
                )
        return requests

    unreferenced_requests = asyncio.run(persist_with_reference_state(False))
    referenced_requests = asyncio.run(persist_with_reference_state(True))
    unverifiable_requests = asyncio.run(persist_with_reference_state(None))
    unreferenced_deletes = [
        request for request in unreferenced_requests
        if request.method == "DELETE"
    ]
    referenced_deletes = [
        request for request in referenced_requests
        if request.method == "DELETE"
    ]
    unverifiable_deletes = [
        request for request in unverifiable_requests
        if request.method == "DELETE"
    ]
    assert len(unreferenced_deletes) == 1
    assert json.loads(unreferenced_deletes[0].content)["prefixes"] == [object_path]
    assert referenced_deletes == []
    assert unverifiable_deletes == []
