"""Opt-in, read-only evaluation against an explicitly selected Neo4j database."""

import asyncio
import os
from pathlib import Path
from typing import Literal, Sequence

import pytest
from dotenv import load_dotenv

from src.graph.embeddings import (
    EmbeddingConfig,
    create_embedding_provider,
)
from src.graph.evaluation import load_evaluation_dataset, run_and_save_evaluation
from src.graph.retrieval import (
    FullTextRetriever,
    HybridRetriever,
    RRFConfig,
    SearchIndexUnavailable,
    VectorRetriever,
    _require_search_indexes_online,
)

pytestmark = pytest.mark.live_retrieval
DATASET_PATH = Path("src/graph/evaluation/legal_corpus_dev_v1.json")


def _live_db():
    load_dotenv(".env")
    if os.getenv("GRAPH_RAG_LIVE_RETRIEVAL") != "1":
        pytest.skip("set GRAPH_RAG_LIVE_RETRIEVAL=1 to run read-only live retrieval tests")
    database = os.getenv("NEO4J_LIVE_DATABASE") or os.getenv("NEO4J_DATABASE")
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
    if not database or any(not os.getenv(key) for key in required):
        pytest.skip("set NEO4J_DATABASE (or NEO4J_LIVE_DATABASE) and Neo4j credentials")
    from src.infrastructure.neo4j_client import DbConnect

    return DbConnect(
        url=os.environ["NEO4J_URI"], username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"], database=database,
    )


class _NoEmbeddingProvider:
    provider = "openai"
    model = "live-retrieval-index-readiness-probe"
    dimensions = 2

    async def embed(self, texts: Sequence[str], *, input_type: Literal["query", "document"]):
        raise AssertionError("embedding must not start before indexes are ONLINE")


def test_live_retrievers_fail_closed_when_required_indexes_are_missing_or_offline():
    async def run():
        db = _live_db()
        config = EmbeddingConfig("openai", _NoEmbeddingProvider.model, 2)
        try:
            try:
                await _require_search_indexes_online(db)
                fulltext_online = True
            except SearchIndexUnavailable:
                fulltext_online = False
            try:
                await _require_search_indexes_online(
                    db, config.vector_index_name, require_fulltext=False
                )
                vector_online = True
            except SearchIndexUnavailable:
                vector_online = False

            if not fulltext_online:
                with pytest.raises(SearchIndexUnavailable):
                    await FullTextRetriever(db).search("retrieval readiness", limit=1)
            if not vector_online:
                with pytest.raises(SearchIndexUnavailable):
                    await VectorRetriever(db, _NoEmbeddingProvider(), config).search(
                        "retrieval readiness", limit=1
                    )
            if not (fulltext_online and vector_online):
                with pytest.raises(SearchIndexUnavailable):
                    await HybridRetriever(
                        VectorRetriever(db, _NoEmbeddingProvider(), config),
                        FullTextRetriever(db),
                    ).search("retrieval readiness", limit=1)
            else:
                pytest.skip("required indexes are ONLINE; full corpus evaluation is separate")
        finally:
            await db.close()

    asyncio.run(run())


class _QueryEmbeddingCache:
    """Avoid duplicate provider calls when vector and hybrid use the same queries."""

    def __init__(self, provider):
        self._provider = provider
        self.provider = provider.provider
        self.model = provider.model
        self.dimensions = provider.dimensions
        self._cache: dict[tuple[str, str], list[float]] = {}

    async def embed(self, texts: Sequence[str], *, input_type: Literal["query", "document"]):
        missing = [text for text in texts if (input_type, text) not in self._cache]
        if missing:
            vectors = await self._provider.embed(missing, input_type=input_type)
            self._cache.update({(input_type, text): vector
                                for text, vector in zip(missing, vectors, strict=True)})
        return [self._cache[(input_type, text)] for text in texts]


def test_live_fulltext_vector_and_hybrid_evaluate_same_frozen_corpus():
    async def run():
        db = _live_db()
        dataset = load_evaluation_dataset(DATASET_PATH)
        try:
            try:
                await _require_search_indexes_online(db)
                config = EmbeddingConfig.from_env()
                await _require_search_indexes_online(db, config.vector_index_name)
            except (SearchIndexUnavailable, ValueError) as exc:
                pytest.skip(f"live evaluation prerequisites are unavailable: {exc}")

            async with db.get_session() as session:
                result = await session.run(
                    "MATCH (d:DOCUMENT) WHERE d.source_sha256 IN $sources "
                    "RETURN d.source_sha256 AS source_sha256, "
                    "d.current_ingestion_run_id AS run_id, "
                    "d.source_page_count AS source_page_count",
                    sources=list(dataset.corpus_snapshot),
                )
                rows = await result.data()
            current = {row["source_sha256"]: row for row in rows}
            if set(current) != set(dataset.corpus_snapshot):
                pytest.skip("selected database does not contain all frozen corpus documents")
            stale = {
                source: (dataset.corpus_snapshot[source], current[source]["run_id"])
                for source in dataset.corpus_snapshot
                if current[source]["run_id"] != dataset.corpus_snapshot[source]
            }
            assert not stale, f"frozen evaluation snapshot is stale: {stale}"

            provider = _QueryEmbeddingCache(create_embedding_provider(config))
            fulltext = FullTextRetriever(db)
            vector = VectorRetriever(db, provider, config)
            hybrid = HybridRetriever(vector, fulltext, RRFConfig())
            strategies = {"fulltext": fulltext, "vector": vector, "hybrid": hybrid}
            metrics = {}
            for name, retriever in strategies.items():
                run_result, artifact = await run_and_save_evaluation(
                    dataset, retriever,
                    artifact_directory=Path("artifacts/graph_evaluation") / name,
                    k=5, abstention_score_threshold=0.01,
                )
                assert artifact.exists()
                assert run_result["dataset_version"] == dataset.version
                assert run_result["corpus_snapshot"] == dict(dataset.corpus_snapshot)
                assert run_result["configuration"]["retrieval"]
                for query_result in run_result["per_query"]:
                    for evidence in query_result["evidence"]:
                        assert evidence["document_sha256"] in dataset.corpus_snapshot
                        assert evidence["ingestion_run_id"] == dataset.corpus_snapshot[
                            evidence["document_sha256"]
                        ]
                        assert evidence["sources"]
                        page_count = current[evidence["document_sha256"]]["source_page_count"]
                        assert all(1 <= source["page_number"] <= page_count
                                   for source in evidence["sources"])
                metrics[name] = run_result["metrics"]
            assert set(metrics) == {"fulltext", "vector", "hybrid"}
        finally:
            await db.close()

    asyncio.run(run())
