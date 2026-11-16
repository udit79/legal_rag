from __future__ import annotations

import json
import asyncio
import hashlib
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.graph import get_graph, get_private_visual_asset, router as graph_router
from src.graph.retrieval import (
    EvidenceBundle,
    ImageSearchEvidence,
    MultimodalSearchResult,
)
from src.ingestion.chunk_contracts import ChunkSource, VisualAssetReference
from src.ingestion.provenance.image_asset_resolver import (
    ResolvedImageAsset,
    SupabaseImageAssetResolver,
)
from src.ingestion.parser_contracts import SourceCoordinates


class _Result:
    def __init__(self, records):
        self._records = records

    def __aiter__(self):
        async def iterate():
            for record in self._records:
                yield record
        return iterate()


class _Session:
    def __init__(self, node_records, edge_records):
        self.node_records = node_records
        self.edge_records = edge_records
        self.calls = []

    async def run(self, query, **params):
        self.calls.append((query, params))
        return _Result(self.node_records if "MATCH (n)" in query else self.edge_records)


class _Db:
    def __init__(self, session):
        self.session = session

    @asynccontextmanager
    async def get_session(self):
        yield self.session


class _Resolver:
    async def resolve(self, run_id, keys):
        return {
            (page, block): ResolvedImageAsset(
                ingestion_run_id=run_id,
                page_number=page,
                block_id=block,
                image_sha256="a" * 64,
                bucket="ingestion-image-crops",
                object_path=f"{run_id}/{page}/{block}.png",
                file_size_bytes=42,
                ocr_status="TEXT_FOUND",
                ocr_text_sha256="b" * 64,
                ocr_text_length=8,
                decision="ADMITTED",
                source_coordinates=SourceCoordinates(
                    x0=1, y0=2, x1=4, y1=6, unit="PIXEL", origin="TOP_LEFT"
                ),
                reading_order=2,
            )
            for page, block in keys
        }


def _request(session, resolver=None):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        db=_Db(session), image_asset_resolver=resolver,
    )))


def test_graph_api_returns_nodes_relationships_and_private_visual_reference():
    evidence = {
        "ingestion_run_id": "run-1",
        "page_number": 3,
        "source_provenance": {
            "extraction_method": "IMAGE_OCR",
            "source_image_block_id": "page-3-block-2",
            "source_image_sha256": "a" * 64,
        },
    }
    node_records = [
        {"properties": {"id": "entity-1", "name": "Regulator",
                        "evidence_references": [json.dumps(evidence)]},
         "labels": ["SEMANTIC_ENTITY", "ORGANIZATION"]},
        {"properties": {"id": "entity-2", "name": "Portal"},
         "labels": ["SEMANTIC_ENTITY", "APPLICATION"]},
    ]
    edges = [{"properties": {"id": "rel-1", "confidence": 0.95},
              "source": "entity-1", "source_properties": node_records[0]["properties"],
              "source_labels": node_records[0]["labels"], "source_run_id": None,
              "target": "entity-2", "target_properties": node_records[1]["properties"],
              "target_labels": node_records[1]["labels"], "target_run_id": None,
              "type": "OWNS"}]
    session = _Session(node_records, edges)

    result = asyncio.run(get_graph(_request(session, _Resolver()), limit=10, offset=0))

    assert [node.id for node in result.nodes] == ["entity-1", "entity-2"]
    assert result.edges[0].model_dump() == {
        "id": "rel-1", "type": "OWNS", "source": "entity-1", "target": "entity-2",
        "properties": {"id": "rel-1", "confidence": 0.95},
    }
    assert len(result.visual_assets) == 1
    asset = result.visual_assets[0]
    assert asset.attached_to == "entity-1"
    assert asset.reference.object_path == "run-1/3/page-3-block-2.png"
    dumped = result.model_dump_json()
    assert "public_url" not in dumped and "signed_url" not in dumped
    assert "image_bytes" not in dumped
    assert session.calls[0][1] == {"limit": 11, "offset": 0}
    assert session.calls[1][1] == {"node_ids": ["entity-1", "entity-2"]}


def test_graph_api_paginates_large_graph():
    session = _Session([
        {"properties": {"id": "a", "name": "A"}, "labels": ["SEMANTIC_ENTITY", "RISK"]},
        {"properties": {"id": "b", "name": "B"}, "labels": ["SEMANTIC_ENTITY", "RISK"]},
    ], [])
    result = asyncio.run(get_graph(_request(session), limit=1, offset=0))
    assert [node.id for node in result.nodes] == ["a"]
    assert result.next_offset == 1


def test_image_delivery_route_returns_only_verified_private_crop_bytes():
    import asyncio

    class ImageResolver(_Resolver):
        async def resolve(self, run_id, keys):
            resolved = await super().resolve(run_id, keys)
            return resolved

        async def download_verified(self, reference):
            assert reference.image_sha256 == "a" * 64
            return b"\x89PNG\r\n\x1a\nimage", "image/png"

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        image_asset_resolver=ImageResolver(),
    )))
    response = asyncio.run(get_private_visual_asset(
        request, "run-1", 3, "page-3-block-2", "a" * 64
    ))
    assert response.body.startswith(b"\x89PNG")
    assert response.media_type == "image/png"
    assert response.headers["cache-control"].startswith("private")
    assert response.headers["x-content-type-options"] == "nosniff"


def _route_app(*, results=(), resolver=None, available=True, error=None):
    app = FastAPI()
    app.include_router(graph_router)
    app.state.image_asset_resolver = resolver
    if available:
        class Retriever:
            async def search_multimodal(self, query, visual_retriever, limit):
                assert query == "bank connection"
                assert limit == 5
                if error:
                    raise error
                return list(results)

        app.state.multimodal_retriever = Retriever()
        app.state.visual_graph_retriever = object()
    return app


def _image_result(channels=("visual_graph",)):
    reference = VisualAssetReference(
        ingestion_run_id="run-1", page_number=3, source_block_id="page-3-block-2",
        image_sha256="a" * 64, bucket="ingestion-image-crops",
        object_path="run-1/3/crop.png", file_size_bytes=42,
    )
    image = ImageSearchEvidence(
        visual_asset=reference,
        image_url=f"/visual-assets/run-1/3/page-3-block-2?image_sha256={'a' * 64}",
        source_coordinates={"x0": 1, "y0": 2, "x1": 4, "y1": 6},
        document_sha256="c" * 64, ingestion_run_id="run-1", page_number=3,
        source_chunk_id="chunk-vlm-1", supporting_description="Bank connects to Portal",
        visual_description_sha256="b" * 64, visual_model_provider="openai",
        visual_model="gpt-4o-mini", visual_prompt_version="visual-v1",
    )
    return MultimodalSearchResult(
        id="image:run-1:3:page-3-block-2", result_type="image", score=0.08,
        channels=channels, image_evidence=image,
    )


def _text_result():
    source = ChunkSource(
        page_number=1, block_id="page-1-block-1", block_reading_order=0,
        block_type="TEXT", source_text_length=18, start_offset=0, end_offset=18,
    )
    evidence = EvidenceBundle(
        chunk_id="chunk-text-1", document_sha256="c" * 64, raw_text="Native source text",
        sources=(source,), ingestion_run_id="run-1", retrieval_channels=("fulltext",),
        fulltext_score=0.9,
    )
    return MultimodalSearchResult(
        id="chunk:chunk-text-1", result_type="text", score=0.08,
        channels=("fulltext",), text_evidence=evidence,
    )


def test_post_graph_search_returns_visual_graph_image_hit():
    with TestClient(_route_app(results=(_image_result(),))) as client:
        response = client.post("/graph/search", json={"query": "bank connection", "limit": 5})
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["result_type"] == "image"
    assert result["image_evidence"]["visual_asset"]["image_sha256"] == "a" * 64
    assert result["image_evidence"]["trust_status"] == "UNVERIFIED_AI_VISUAL_OBSERVATION"


def test_post_graph_search_returns_image_when_a_vlm_chunk_is_the_text_retrieval_hit():
    result = _image_result(channels=("vector", "fulltext"))
    with TestClient(_route_app(results=(result,))) as client:
        response = client.post("/graph/search", json={"query": "bank connection", "limit": 5})
    assert response.status_code == 200
    output = response.json()["results"][0]
    assert output["result_type"] == "image"
    assert output["channels"] == ["vector", "fulltext"]
    assert output["image_evidence"]["supporting_description"]


def test_post_graph_search_returns_mixed_text_and_image_results():
    with TestClient(_route_app(results=(_text_result(), _image_result()))) as client:
        response = client.post("/graph/search", json={"query": "bank connection", "limit": 5})
    assert response.status_code == 200
    assert [item["result_type"] for item in response.json()["results"]] == ["text", "image"]


def test_post_graph_search_handles_missing_configuration_and_retrieval_errors():
    with TestClient(_route_app(available=False)) as client:
        unavailable = client.post("/graph/search", json={"query": "bank connection"})
    with TestClient(_route_app(error=RuntimeError("index missing"))) as client:
        failed = client.post("/graph/search", json={"query": "bank connection"})
    assert unavailable.status_code == 503
    assert failed.status_code == 503


def test_visual_asset_route_checks_requested_hash_and_streamed_length():
    body = b"\x89PNG\r\n\x1a\nexample"
    asset = ResolvedImageAsset(
        ingestion_run_id="run-1", page_number=3, block_id="page-3-block-2",
        image_sha256=hashlib.sha256(body).hexdigest(),
        bucket="ingestion-image-crops", object_path="run-1/3/crop.png",
        file_size_bytes=len(body), ocr_status="TEXT_FOUND", ocr_text_sha256="b" * 64,
        ocr_text_length=8, decision="ADMITTED",
        source_coordinates=SourceCoordinates(
            x0=1, y0=2, x1=4, y1=6, unit="PIXEL", origin="TOP_LEFT"
        ), reading_order=2,
    )

    class VerifiedResolver:
        async def resolve(self, run_id, keys):
            return {(3, "page-3-block-2"): asset}

        async def download_verified(self, reference):
            assert reference.image_sha256 == asset.image_sha256
            assert len(body) == reference.file_size_bytes
            return body, "image/png"

    with TestClient(_route_app(resolver=VerifiedResolver(), available=False)) as client:
        served = client.get(
            f"/visual-assets/run-1/3/page-3-block-2?image_sha256={asset.image_sha256}"
        )
        mismatch = client.get(
            f"/visual-assets/run-1/3/page-3-block-2?image_sha256={'f' * 64}"
        )
    assert served.status_code == 200
    assert served.content == body
    assert served.headers["content-length"] == str(len(body))
    assert hashlib.sha256(served.content).hexdigest() == asset.image_sha256
    assert mismatch.status_code == 404


def test_visual_asset_route_returns_404_for_missing_asset_metadata():
    class MissingResolver:
        async def resolve(self, run_id, keys):
            return {}

    with TestClient(_route_app(resolver=MissingResolver(), available=False)) as client:
        response = client.get(
            f"/visual-assets/run-1/3/page-3-block-2?image_sha256={'a' * 64}"
        )
    assert response.status_code == 404


def test_visual_asset_endpoint_hash_checks_private_supabase_download():
    import httpx

    body = b"\x89PNG\r\n\x1a\nverified crop"
    digest = hashlib.sha256(body).hexdigest()
    metadata = [{
        "ingestion_run_id": "run-live-test", "page_number": 4,
        "block_id": "page-4-block-2", "image_sha256": digest,
        "image_bucket": "ingestion-image-crops", "image_object_path": "run-live-test/crop.png",
        "image_file_size_bytes": len(body), "ocr_status": "TEXT_FOUND",
        "ocr_text_sha256": "b" * 64, "ocr_text_length": 12,
        "decision": "ADMITTED", "source_coordinates": {
            "x0": 1, "y0": 2, "x1": 5, "y1": 6,
            "unit": "PIXEL", "origin": "TOP_LEFT",
        }, "reading_order": 2,
    }]

    async def serve(request):
        if request.url.path == "/rest/v1/image_screening_results":
            return httpx.Response(200, json=metadata)
        if request.url.path.endswith("/storage/v1/object/authenticated/ingestion-image-crops/run-live-test/crop.png"):
            return httpx.Response(200, content=body)
        return httpx.Response(404)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(serve))
    resolver = SupabaseImageAssetResolver("https://supabase.test", "unit-key", async_client)
    with TestClient(_route_app(resolver=resolver, available=False)) as client:
        response = client.get(
            f"/visual-assets/run-live-test/4/page-4-block-2?image_sha256={digest}"
        )
    asyncio.run(async_client.aclose())
    assert response.status_code == 200
    assert response.content == body
    assert hashlib.sha256(response.content).hexdigest() == digest
    assert response.headers["content-length"] == str(len(body))


def test_visual_asset_endpoint_rejects_corrupt_private_download():
    import httpx

    body = b"\x89PNG\r\n\x1a\ncorrect"
    corrupt = b"\x89PNG\r\n\x1a\nwrong!!"
    digest = hashlib.sha256(body).hexdigest()
    metadata = [{
        "ingestion_run_id": "run-corrupt", "page_number": 4,
        "block_id": "page-4-block-2", "image_sha256": digest,
        "image_bucket": "ingestion-image-crops", "image_object_path": "run-corrupt/crop.png",
        "image_file_size_bytes": len(body), "ocr_status": "TEXT_FOUND",
        "ocr_text_sha256": "b" * 64, "ocr_text_length": 12,
        "decision": "ADMITTED", "source_coordinates": {
            "x0": 1, "y0": 2, "x1": 5, "y1": 6,
            "unit": "PIXEL", "origin": "TOP_LEFT",
        }, "reading_order": 2,
    }]

    async def serve(request):
        if request.url.path == "/rest/v1/image_screening_results":
            return httpx.Response(200, json=metadata)
        return httpx.Response(200, content=corrupt)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(serve))
    resolver = SupabaseImageAssetResolver("https://supabase.test", "unit-key", async_client)
    with TestClient(_route_app(resolver=resolver, available=False)) as client:
        response = client.get(
            f"/visual-assets/run-corrupt/4/page-4-block-2?image_sha256={digest}"
        )
    asyncio.run(async_client.aclose())
    assert response.status_code == 503

