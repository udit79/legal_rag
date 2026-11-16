from __future__ import annotations

import json
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from src.api.graph import get_graph, get_private_visual_asset
from src.ingestion.provenance.image_asset_resolver import ResolvedImageAsset
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

