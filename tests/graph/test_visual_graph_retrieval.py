import asyncio
import hashlib

import pytest

from src.graph.retrieval import (
    EvidenceBundle,
    HybridRetriever,
    RetrievalHit,
    RRFConfig,
    SearchIndexUnavailable,
    VisualGraphRetriever,
)
from src.ingestion.chunk_contracts import ChunkSource
from src.ingestion.parser_contracts import SourceCoordinates
from src.ingestion.provenance.image_asset_resolver import ResolvedImageAsset


RUN = "run-vis"
DOC = "a" * 64
IMAGE = "b" * 64
DESC = "A diagram shows Bank connected to a core system."
DESC_SHA = hashlib.sha256(DESC.encode()).hexdigest()
BLOCK = "page-1-block-2"


def _observation():
    return {
        "id": "visual-observation:" + "c" * 64,
        "source_sha256": DOC,
        "ingestion_run_id": RUN,
        "chunking_policy_version": "cpcr-v1",
        "source_chunk_id": "chunk-vlm-1",
        "page_number": 1,
        "source_image_block_id": BLOCK,
        "image_sha256": IMAGE,
        "source_coordinates": SourceCoordinates(
            x0=1, y0=2, x1=3, y1=4, unit="PDF_POINT", origin="TOP_LEFT"
        ).model_dump(mode="json"),
        "reading_order": 2,
        "description_text": DESC,
        "description_sha256": DESC_SHA,
        "description_excerpt_sha256": DESC_SHA,
        "vlm_provider": "openai",
        "vlm_model": "gpt-4o-mini",
        "prompt_version": "visual-description-v1",
        "extraction_provider": "openai",
        "extraction_model": "visual-extractor-v1",
        "trust_status": "UNVERIFIED_AI_VISUAL_OBSERVATION",
    }


def _asset():
    return ResolvedImageAsset(
        ingestion_run_id=RUN, page_number=1, block_id=BLOCK, image_sha256=IMAGE,
        bucket="ingestion-image-crops", object_path=f"{RUN}/crop.png", file_size_bytes=30,
        ocr_status="NO_TEXT", ocr_text_sha256=None, ocr_text_length=0,
        visual_description_sha256=DESC_SHA, visual_description_text=DESC,
        visual_description_provider="openai", visual_description_model="gpt-4o-mini",
        visual_prompt_version="visual-description-v1", decision=None,
        source_coordinates=SourceCoordinates(
            x0=1, y0=2, x1=3, y1=4, unit="PDF_POINT", origin="TOP_LEFT"
        ), reading_order=2,
    )


class _Result:
    def __init__(self, rows):
        self.rows = rows

    async def data(self):
        return self.rows


class _Session:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, query, **params):
        self.db.queries.append(query)
        if "SHOW INDEXES" in query:
            return _Result(self.db.indexes)
        if "SEARCH hit IN" in query:
            return _Result([{
                "observation": _observation(),
                "candidate_entities": [{"name": "Bank", "category": "ORGANIZATION"}],
                "visual_relationships": [],
                "matched_semantic_entities": [{"id": "organization:bank", "name": "Bank"}],
                "score": 0.75,
            }])
        raise AssertionError(query)


class _Db:
    def __init__(self, online=True):
        self.indexes = ([{"name": "visual_knowledge_fulltext", "type": "FULLTEXT", "state": "ONLINE"}]
                        if online else [])
        self.queries = []

    def get_session(self):
        return _Session(self)


class _Resolver:
    async def resolve(self, run_id, keys):
        assert run_id == RUN and keys == ((1, BLOCK),)
        return {(1, BLOCK): _asset()}


def test_visual_graph_retriever_requires_own_online_index_and_checks_provenance():
    db = _Db()
    hits = asyncio.run(VisualGraphRetriever(db, image_asset_resolver=_Resolver()).search("core system"))
    assert len(hits) == 1
    hit = hits[0]
    assert hit.trust_status == "UNVERIFIED_AI_VISUAL_OBSERVATION"
    assert hit.visual_asset.image_sha256 == IMAGE
    assert hit.candidate_entities[0]["name"] == "Bank"
    assert "SEARCH hit IN" in db.queries[-1]
    assert all("db.index.vector.query" not in query and "db.index.fulltext.query" not in query
               for query in db.queries)
    with pytest.raises(SearchIndexUnavailable, match="visual_knowledge_fulltext"):
        asyncio.run(VisualGraphRetriever(_Db(False), image_asset_resolver=_Resolver()).search("bank"))


def test_multimodal_hybrid_fuses_visual_graph_by_rank_and_returns_image_result():
    source = ChunkSource(
        page_number=2, block_id="page-2-block-1", block_reading_order=1,
        block_type="TEXT", source_text_length=12, start_offset=0, end_offset=12,
    )
    hit = RetrievalHit(
        chunk_id="chunk-text", document_sha256=DOC, raw_text="Bank system",
        sources=(source,), ingestion_run_id=RUN, channel="vector", score=0.92,
    )

    class FakeVector:
        _db = _Db()
        _db.indexes = [
            {"name": "chunk_retrieval_text_fulltext", "type": "FULLTEXT", "state": "ONLINE"},
            {"name": "vector-index", "type": "VECTOR", "state": "ONLINE"},
        ]
        _index_name = "vector-index"
        _image_asset_resolver = None
        _config = type("Config", (), {"provider": "openai", "model": "embed", "dimensions": 3})()
        _candidate_multiplier = 5
        async def _search(self, query, limit): return [hit]

    class FakeFulltext:
        _db = FakeVector._db
        _index_name = "chunk_retrieval_text_fulltext"
        _image_asset_resolver = None
        _candidate_multiplier = 5
        async def _search(self, query, limit): return []

    class FakeVisual:
        async def search(self, query, limit):
            from src.graph.retrieval import VisualGraphEvidence
            return [VisualGraphEvidence(
                result_id=_observation()["id"], document_sha256=DOC, ingestion_run_id=RUN,
                chunking_policy_version="cpcr-v1", page_number=1, source_chunk_id="chunk-vlm-1",
                source_image_block_id=BLOCK, image_sha256=IMAGE, description_text=DESC,
                source_coordinates=_asset().source_coordinates.model_dump(mode="json"),
                description_sha256=DESC_SHA, description_excerpt_sha256=DESC_SHA,
                vlm_provider="openai", vlm_model="gpt-4o-mini", prompt_version="visual-description-v1",
                extraction_provider="openai", extraction_model="visual-extractor-v1",
                trust_status="UNVERIFIED_AI_VISUAL_OBSERVATION", candidate_entities=(),
                visual_relationships=(), matched_semantic_entities=(), score=0.75,
                visual_asset=_asset().reference,
            )]

    results = asyncio.run(HybridRetriever(
        FakeVector(), FakeFulltext(), RRFConfig(rank_constant=10)
    ).search_multimodal("bank controls", FakeVisual(), limit=5, visual_weight=2))
    assert [item.result_type for item in results] == ["image", "text"]
    assert results[0].image_evidence.image_url.endswith(f"?image_sha256={IMAGE}")
    assert "visual_graph" in results[0].channels


def test_vlm_text_results_become_original_image_results_without_losing_mixed_text():
    from src.ingestion.parser_contracts import canonical_visual_block_id

    vlm_source = ChunkSource(
        page_number=1,
        block_id=canonical_visual_block_id(
            1, BLOCK, "openai", "gpt-4o-mini", "visual-description-v1", DESC_SHA
        ),
        block_reading_order=2, block_derived_order=2, block_type="IMAGE",
        source_text_length=len(DESC), start_offset=0, end_offset=len(DESC),
        extraction_method="IMAGE_VLM", source_image_block_id=BLOCK,
        source_image_sha256=IMAGE, visual_description_sha256=DESC_SHA,
        visual_model_provider="openai", visual_model="gpt-4o-mini",
        visual_prompt_version="visual-description-v1",
    )
    native_text = "The policy is current. "
    native_source = ChunkSource(
        page_number=1, block_id="page-1-block-1", block_reading_order=1,
        block_type="TEXT", source_text_length=len(native_text), start_offset=0,
        end_offset=len(native_text),
    )
    mixed_raw = native_text + DESC

    class Resolver:
        async def resolve(self, run_id, keys):
            return {(1, BLOCK): _asset()}

    class FakeVector:
        _db = _Db()
        _db.indexes = [
            {"name": "chunk_retrieval_text_fulltext", "type": "FULLTEXT", "state": "ONLINE"},
            {"name": "vector-index", "type": "VECTOR", "state": "ONLINE"},
        ]
        _index_name = "vector-index"
        _image_asset_resolver = Resolver()
        _config = type("Config", (), {"provider": "openai", "model": "embed", "dimensions": 3})()
        _candidate_multiplier = 5
        async def _search(self, query, limit):
            return [RetrievalHit(
                chunk_id="mixed-chunk", document_sha256=DOC, raw_text=mixed_raw,
                sources=(native_source, vlm_source), ingestion_run_id=RUN,
                channel="vector", score=0.9,
            )]

    class FakeFulltext:
        _db = FakeVector._db
        _index_name = "chunk_retrieval_text_fulltext"
        _image_asset_resolver = FakeVector._image_asset_resolver
        _candidate_multiplier = 5
        async def _search(self, query, limit): return []

    class NoGraph:
        async def search(self, query, limit): return []

    results = asyncio.run(HybridRetriever(
        FakeVector(), FakeFulltext(), RRFConfig()
    ).search_multimodal("policy controls", NoGraph()))
    assert {item.result_type for item in results} == {"image", "text"}
    image = next(item.image_evidence for item in results if item.result_type == "image")
    text = next(item.text_evidence for item in results if item.result_type == "text")
    assert image.supporting_description == DESC
    assert image.visual_asset.image_sha256 == IMAGE
    assert DESC not in text.raw_text
    assert text.raw_text == "The policy is current."


def test_vlm_text_results_become_original_image_results_without_losing_mixed_text():
    from src.ingestion.parser_contracts import canonical_visual_block_id

    vlm_source = ChunkSource(
        page_number=1,
        block_id=canonical_visual_block_id(
            1, BLOCK, "openai", "gpt-4o-mini", "visual-description-v1", DESC_SHA
        ),
        block_reading_order=2, block_derived_order=2, block_type="IMAGE",
        source_text_length=len(DESC), start_offset=0, end_offset=len(DESC),
        extraction_method="IMAGE_VLM", source_image_block_id=BLOCK,
        source_image_sha256=IMAGE, visual_description_sha256=DESC_SHA,
        visual_model_provider="openai", visual_model="gpt-4o-mini",
        visual_prompt_version="visual-description-v1",
    )
    native_text = "The policy is current. "
    native_source = ChunkSource(
        page_number=1, block_id="page-1-block-1", block_reading_order=1,
        block_type="TEXT", source_text_length=len(native_text), start_offset=0,
        end_offset=len(native_text),
    )
    mixed_raw = native_text + DESC

    class Resolver:
        async def resolve(self, run_id, keys):
            return {(1, BLOCK): _asset()}

    class FakeVector:
        _db = _Db()
        _db.indexes = [
            {"name": "chunk_retrieval_text_fulltext", "type": "FULLTEXT", "state": "ONLINE"},
            {"name": "vector-index", "type": "VECTOR", "state": "ONLINE"},
        ]
        _index_name = "vector-index"
        _image_asset_resolver = Resolver()
        _config = type("Config", (), {"provider": "openai", "model": "embed", "dimensions": 3})()
        _candidate_multiplier = 5
        async def _search(self, query, limit):
            return [RetrievalHit(
                chunk_id="mixed-chunk", document_sha256=DOC, raw_text=mixed_raw,
                sources=(native_source, vlm_source), ingestion_run_id=RUN,
                channel="vector", score=0.9,
            )]

    class FakeFulltext:
        _db = FakeVector._db
        _index_name = "chunk_retrieval_text_fulltext"
        _image_asset_resolver = FakeVector._image_asset_resolver
        _candidate_multiplier = 5
        async def _search(self, query, limit): return []

    class NoGraph:
        async def search(self, query, limit): return []

    results = asyncio.run(HybridRetriever(
        FakeVector(), FakeFulltext(), RRFConfig()
    ).search_multimodal("policy controls", NoGraph()))
    assert {item.result_type for item in results} == {"image", "text"}
    image = next(item.image_evidence for item in results if item.result_type == "image")
    text = next(item.text_evidence for item in results if item.result_type == "text")
    assert image.supporting_description == DESC
    assert image.visual_asset.image_sha256 == IMAGE
    assert DESC not in text.raw_text
    assert text.raw_text == "The policy is current."
