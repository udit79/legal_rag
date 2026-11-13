"""Mocked validation tests for the read-only retrieval smoke command."""

import asyncio
import json
import math

import pytest

from scripts.retrieval_smoke import (
    _validate_bundle,
    _validate_current_provenance,
)
from src.graph.embeddings import EmbeddingConfig
from src.graph.retrieval import FullTextRetriever, HybridRetriever, SearchIndexUnavailable, VectorRetriever
from src.graph.retrieval import EvidenceBundle
from src.ingestion.chunk_contracts import ChunkSource

DOC = "a" * 64
SOURCE = ChunkSource(
    page_number=1, block_id="page-1-block-0", block_reading_order=0,
    block_type="TEXT", source_text_length=12, start_offset=0, end_offset=12,
)


def _bundle(strategy: str, **changes):
    channel = {"fulltext": ("fulltext",), "vector": ("vector",),
               "hybrid": ("vector", "fulltext")} [strategy]
    values = {
        "chunk_id": "chunk-1", "document_sha256": DOC, "raw_text": "raw evidence",
        "sources": (SOURCE,), "ingestion_run_id": "run-1",
        "retrieval_channels": channel,
        "vector_score": 0.8 if strategy != "fulltext" else None,
        "fulltext_score": 1.4 if strategy != "vector" else None,
        "rrf_score": 0.03 if strategy == "hybrid" else None,
    }
    values.update(changes)
    return EvidenceBundle(**values)


@pytest.mark.parametrize("strategy", ["fulltext", "vector", "hybrid"])
def test_strategy_bundle_validation_preserves_native_and_rrf_score_contract(strategy):
    evidence = _bundle(strategy)
    _validate_bundle(strategy, evidence)
    assert math.isfinite(evidence.abstention_score)
    assert evidence.abstention_score_type == {
        "fulltext": "fulltext_relevance_score",
        "vector": "vector_relevance_score",
        "hybrid": "rrf_score",
    }[strategy]


@pytest.mark.parametrize("strategy, change", [
    ("fulltext", {"vector_score": 0.4}),
    ("vector", {"rrf_score": 0.4}),
    ("hybrid", {"rrf_score": None}),
])
def test_strategy_bundle_validation_rejects_mislabelled_scores(strategy, change):
    with pytest.raises(ValueError):
        _validate_bundle(strategy, _bundle(strategy, **change))


class _Rows:
    def __init__(self, data):
        self._data = data

    async def data(self):
        return self._data


class _Session:
    def __init__(self, row):
        self.row = row

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def run(self, query, **params):
        assert params["document_hashes"] == [DOC]
        assert params["chunk_ids"] == ["chunk-1"]
        return _Rows([self.row])


class _Db:
    def __init__(self, row):
        self.row = row

    def get_session(self):
        return _Session(self.row)


def _provenance_row(**changes):
    row = {
        "chunk_id": "chunk-1", "document_sha256": DOC,
        "ingestion_run_id": "run-1", "current_policy": "policy-v1",
        "chunk_policy": "policy-v1", "is_current": True,
        "raw_text": "raw evidence", "sources_json": json.dumps([SOURCE.model_dump()]),
    }
    row.update(changes)
    return row


def test_current_provenance_validation_accepts_matching_document_run_and_source_coordinates():
    asyncio.run(_validate_current_provenance(_Db(_provenance_row()), [_bundle("fulltext")]))


@pytest.mark.parametrize("change", [
    {"ingestion_run_id": "old-run"},
    {"current_policy": "policy-v2"},
    {"is_current": False},
    {"raw_text": "different raw text"},
    {"sources_json": "[]"},
])
def test_current_provenance_validation_rejects_stale_or_missing_source_state(change):
    with pytest.raises(ValueError):
        asyncio.run(_validate_current_provenance(
            _Db(_provenance_row(**change)), [_bundle("fulltext")]
        ))


def test_current_provenance_validation_rejects_missing_graph_identity():
    with pytest.raises(ValueError, match="not linked to a current document chunk"):
        class _EmptyRows(_Session):
            async def run(self, query, **params):
                return _Rows([])

        class _EmptyDb:
            def get_session(self):
                return _EmptyRows(None)

        asyncio.run(_validate_current_provenance(_EmptyDb(), [_bundle("fulltext")]))


def test_retrievers_return_normalized_empty_results_when_indexes_are_online():
    class _Result:
        async def data(self):
            return []

    class _RetrievalSession(_Session):
        async def run(self, query, **params):
            if "SHOW INDEXES" in query:
                return _Rows([
                    {"name": name,
                     "type": "VECTOR" if name.startswith("chunk_embedding_") else "FULLTEXT",
                     "state": "ONLINE"}
                    for name in params["names"]
                ])
            return _Result()

    class _RetrievalDb:
        def get_session(self):
            return _RetrievalSession(None)

    class _Provider:
        provider = "openai"
        model = "unit-model"
        dimensions = 2

        async def embed(self, texts, *, input_type):
            return [[1.0, 0.0] for _ in texts]

    async def run():
        db = _RetrievalDb()
        config = EmbeddingConfig("openai", _Provider.model, 2)
        fulltext = FullTextRetriever(db)
        vector = VectorRetriever(db, _Provider(), config)
        hybrid = HybridRetriever(vector, fulltext)
        assert await fulltext.search("query", limit=2) == []
        assert await vector.search("query", limit=2) == []
        assert await hybrid.search("query", limit=2) == []

    asyncio.run(run())
