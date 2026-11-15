import asyncio
import hashlib

import httpx
import pytest

from src.graph.retrieval import EvidenceBundle, FullTextRetriever
from src.ingestion.chunk_contracts import Chunk, ChunkSource, VisualAssetReference
from src.ingestion.provenance.image_asset_resolver import (
    ImageAssetResolutionError,
    ResolvedImageAsset,
    SupabaseImageAssetResolver,
)
from src.graph.retrieval import (
    VisualAssetResolutionUnavailable,
    resolve_visual_asset_references,
)
from src.ingestion.parser_contracts import SourceCoordinates


RUN = "run-current"
BLOCK = "page-4-block-8"
OCR_BLOCK = f"page-4-block-ocr-{hashlib.sha256(b'key').hexdigest()}"
COORDINATES = {
    "x0": 1.0, "y0": 2.0, "x1": 30.0, "y1": 40.0,
    "unit": "PDF_POINT", "origin": "TOP_LEFT",
}


def _chunk() -> Chunk:
    text = "OCR clause"
    return Chunk(
        index=0,
        text=text,
        sources=(ChunkSource(
            page_number=4,
            block_id=OCR_BLOCK,
            block_reading_order=9,
            block_type="IMAGE",
            source_text_length=len(text),
            start_offset=0,
            end_offset=len(text),
            extraction_method="IMAGE_OCR",
            source_image_block_id=BLOCK,
            ocr_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        ),),
    )


def _asset_row(run_id=RUN, decision="ADMITTED"):
    return {
        "ingestion_run_id": run_id,
        "page_number": 4,
        "block_id": BLOCK,
        "image_sha256": "a" * 64,
        "image_bucket": "ingestion-image-crops",
        "image_object_path": f"{run_id}/crop.png",
        "image_file_size_bytes": 128,
        "ocr_status": "TEXT_FOUND",
        "ocr_text_sha256": hashlib.sha256(b"OCR clause").hexdigest(),
        "ocr_text_length": len("OCR clause"),
        "decision": decision,
        "source_coordinates": COORDINATES,
        "reading_order": 8,
    }


def _resolved_asset(run_id=RUN):
    row = _asset_row(run_id=run_id)
    return ResolvedImageAsset(
        ingestion_run_id=run_id,
        page_number=row["page_number"],
        block_id=row["block_id"],
        image_sha256=row["image_sha256"],
        bucket=row["image_bucket"],
        object_path=row["image_object_path"],
        file_size_bytes=row["image_file_size_bytes"],
        ocr_status=row["ocr_status"],
        ocr_text_sha256=row["ocr_text_sha256"],
        ocr_text_length=row["ocr_text_length"],
        decision=row["decision"],
        source_coordinates=COORDINATES,
        reading_order=row["reading_order"],
    )


def test_resolver_reads_exact_composite_key_and_retains_all_asset_metadata():
    calls = []

    async def handle(request):
        calls.append(request)
        return httpx.Response(200, json=[_asset_row()])

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            resolver = SupabaseImageAssetResolver("https://db.test", "service", client)
            return await resolver.resolve(RUN, ((4, BLOCK),))

    resolved = asyncio.run(run())[(4, BLOCK)]
    assert calls[0].url.path == "/rest/v1/image_screening_results"
    assert calls[0].url.params["ingestion_run_id"] == f"eq.{RUN}"
    assert calls[0].url.params["page_number"] == "eq.4"
    assert calls[0].url.params["block_id"] == f"eq.{BLOCK}"
    assert resolved.image_sha256 == "a" * 64
    assert resolved.bucket == "ingestion-image-crops"
    assert resolved.object_path == f"{RUN}/crop.png"
    assert resolved.file_size_bytes == 128
    assert resolved.ocr_status == "TEXT_FOUND"
    assert resolved.decision == "ADMITTED"
    assert resolved.reading_order == 8
    assert resolved.source_coordinates == SourceCoordinates.model_validate(COORDINATES)


def test_resolver_fails_closed_for_missing_or_quarantined_metadata():
    async def verify(response_json):
        async def handle(request):
            return httpx.Response(200, json=response_json)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            resolver = SupabaseImageAssetResolver("https://db.test", "service", client)
            return await resolver.resolve(RUN, ((4, BLOCK),))

    with pytest.raises(ImageAssetResolutionError, match="exactly one"):
        asyncio.run(verify([]))
    with pytest.raises(ImageAssetResolutionError, match="inconsistent"):
        asyncio.run(verify([_asset_row(decision="QUARANTINED")]))


def test_fulltext_retrieval_resolves_asset_after_neo4j_session_closes():
    source = _chunk().sources[0]
    resolved_asset = _resolved_asset()

    class Result:
        def __init__(self, rows):
            self.rows = rows

        async def data(self):
            return self.rows

    class Session:
        def __init__(self, db):
            self.db = db

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.db.closed = True

        async def run(self, query, **params):
            if "SHOW INDEXES" in query:
                return Result([{
                    "name": "chunk_retrieval_text_fulltext",
                    "type": "FULLTEXT",
                    "state": "ONLINE",
                }])
            return Result([{
                "chunk_id": "chunk-ocr",
                "document_sha256": "b" * 64,
                "raw_text": "OCR clause",
                "sources_json": [source.model_dump(mode="json", exclude_defaults=True)],
                "ingestion_run_id": RUN,
                "score": 1.25,
            }])

    class Db:
        closed = False

        def get_session(self):
            return Session(self)

    class Resolver:
        async def resolve(self, run_id, keys):
            assert run_id == RUN and keys == ((4, BLOCK),)
            assert db.closed is True
            return {(4, BLOCK): resolved_asset}

    db = Db()
    evidence = asyncio.run(
        FullTextRetriever(db, image_asset_resolver=Resolver()).search("OCR clause")
    )
    assert evidence[0].visual_assets == (resolved_asset.reference,)


def test_chunk_evidence_carries_private_asset_pointer_and_run_version():
    asset = _resolved_asset()
    source = _chunk().sources[0]
    assert source.source_image_block_id == BLOCK
    assert asset.reference == VisualAssetReference(
        ingestion_run_id=RUN,
        page_number=4,
        source_block_id=BLOCK,
        image_sha256="a" * 64,
        bucket="ingestion-image-crops",
        object_path=f"{RUN}/crop.png",
        file_size_bytes=128,
    )
    evidence = EvidenceBundle(
        chunk_id="chunk-ocr",
        document_sha256="b" * 64,
        raw_text="OCR clause",
        sources=(source,),
        ingestion_run_id=RUN,
        retrieval_channels=("fulltext",),
        fulltext_score=0.5,
    )
    unresolved_evidence = evidence
    class Resolver:
        async def resolve(self, run_id, keys):
            assert run_id == RUN and keys == ((4, BLOCK),)
            return {(4, BLOCK): asset}

    evidence = asyncio.run(resolve_visual_asset_references([evidence], Resolver()))[0]
    assert evidence.visual_assets == (asset.reference,)
    assert evidence.visual_assets[0].object_path == f"{RUN}/crop.png"
    assert "https://" not in evidence.model_dump_json()

    with pytest.raises(ValueError, match="does not match evidence provenance"):
        EvidenceBundle(
            chunk_id="chunk-ocr",
            document_sha256="b" * 64,
            raw_text="OCR clause",
            sources=(source,),
            ingestion_run_id="different-run",
            visual_assets=evidence.visual_assets,
            retrieval_channels=("fulltext",),
            fulltext_score=0.5,
        )

    with pytest.raises(VisualAssetResolutionUnavailable, match="configured Supabase"):
        asyncio.run(resolve_visual_asset_references([unresolved_evidence], None))


def test_evidence_asset_resolution_is_replay_stable_and_run_versioned():
    first = _resolved_asset()
    base = EvidenceBundle(
        chunk_id="chunk-ocr", document_sha256="b" * 64,
        raw_text="OCR clause", sources=_chunk().sources,
        ingestion_run_id=RUN, retrieval_channels=("fulltext",), fulltext_score=0.5,
    )
    class FirstResolver:
        async def resolve(self, run_id, keys):
            return {(4, BLOCK): first}
    first_result = asyncio.run(resolve_visual_asset_references([base], FirstResolver()))
    replay = asyncio.run(resolve_visual_asset_references([base], FirstResolver()))
    assert replay == first_result

    next_run = "run-next"
    next_asset = _resolved_asset(run_id=next_run)
    next_base = base.model_copy(update={"ingestion_run_id": next_run})
    class NextResolver:
        async def resolve(self, run_id, keys):
            return {(4, BLOCK): next_asset}
    versioned = asyncio.run(resolve_visual_asset_references([next_base], NextResolver()))
    assert versioned[0].visual_assets[0].ingestion_run_id == next_run
    assert versioned[0].visual_assets[0].object_path != first_result[0].visual_assets[0].object_path


def test_private_visual_asset_reference_rejects_public_urls():
    with pytest.raises(ValueError, match="private relative object path"):
        VisualAssetReference(
            ingestion_run_id=RUN,
            page_number=4,
            source_block_id=BLOCK,
            image_sha256="a" * 64,
            bucket="ingestion-image-crops",
            object_path="https://storage.example/crop.png",
            file_size_bytes=128,
        )
