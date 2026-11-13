"""Focused unit coverage for evaluation artifacts and retrieval result metrics."""

import asyncio
import json
from pathlib import Path

import pytest

from src.graph.embeddings import EmbeddingConfig
from src.graph.evaluation import (
    AnswerCitation,
    EvaluationEvidenceReference,
    FrozenEvaluationDataset,
    run_evaluation,
    run_and_save_evaluation,
)
from src.graph.retrieval import (
    EvidenceBundle,
    FullTextRetriever,
    HybridRetriever,
    VectorRetriever,
)
from src.ingestion.chunk_contracts import ChunkSource

DOC = "a" * 64


def _evidence() -> EvidenceBundle:
    return EvidenceBundle(
        chunk_id="gold-chunk", document_sha256=DOC, raw_text="Raw source evidence.",
        sources=(ChunkSource(
            page_number=1, block_id="page-1-block-0", block_reading_order=0,
            block_type="TEXT", source_text_length=21, start_offset=0, end_offset=21,
        ),),
        ingestion_run_id="run-1", retrieval_channels=("fulltext",),
        fulltext_score=1.0, rrf_score=1.0,
    )


class _StrategyRetriever:
    def __init__(self, strategy: str):
        self.configuration = {"strategy": strategy, "test_setting": "stable"}

    async def search(self, query: str, limit: int):
        return [_evidence()] if query == "answerable query" else []


def test_evaluation_artifacts_keep_dataset_snapshot_strategy_and_query_metrics(tmp_path: Path):
    dataset = FrozenEvaluationDataset(
        dataset_id="unit-evaluation", version="v7", synthetic=False,
        annotation_source="unit-test labels",
        corpus_snapshot={DOC: "run-1"},
        queries=(
            {"query_id": "known", "query": "answerable query",
             "relevant_chunk_ids": ["gold-chunk"], "expected_document_ids": [DOC],
             "unanswerable": False},
            {"query_id": "unknown", "query": "unanswerable query",
             "relevant_chunk_ids": [], "expected_document_ids": [], "unanswerable": True},
        ),
    )

    async def run():
        outputs = {}
        for strategy in ("fulltext", "vector", "hybrid"):
            retriever = _StrategyRetriever(strategy)
            result, artifact = await run_and_save_evaluation(
                dataset, retriever, artifact_directory=tmp_path / strategy,
                k=2, abstention_score_threshold=0.01,
            )
            stored = json.loads(artifact.read_text(encoding="utf-8"))
            assert stored["dataset_version"] == "v7"
            assert stored["corpus_snapshot"] == {DOC: "run-1"}
            assert stored["configuration"]["retrieval"] == retriever.configuration
            assert len(stored["per_query"]) == 2
            assert stored["per_query"][0]["recall_at_k"] == 1.0
            assert stored["per_query"][1]["abstained"] is True
            outputs[strategy] = result["metrics"]
        assert len({tuple(sorted(metrics.items())) for metrics in outputs.values()}) == 1

    asyncio.run(run())


def test_evaluation_rejects_retrieved_evidence_from_stale_corpus_snapshot():
    dataset = FrozenEvaluationDataset(
        dataset_id="stale-corpus", version="v1", synthetic=False,
        annotation_source="unit-test labels", corpus_snapshot={DOC: "run-current"},
        queries=({"query_id": "q", "query": "answerable query",
                  "relevant_chunk_ids": ["gold-chunk"], "expected_document_ids": [DOC],
                  "unanswerable": False},),
    )

    class _StaleRetriever:
        async def search(self, query: str, limit: int):
            return [_evidence()]

    async def run():
        from src.graph.evaluation import run_evaluation

        with pytest.raises(ValueError, match="frozen corpus snapshot"):
            await run_evaluation(
                dataset, _StaleRetriever(), abstention_score_threshold=0.01,
            )

    asyncio.run(run())


def test_fulltext_vector_and_hybrid_normalize_and_evaluate_both_query_types(tmp_path: Path):
    source = ChunkSource(
        page_number=1, block_id="page-1-block-0", block_reading_order=0,
        block_type="TEXT", source_text_length=21, start_offset=0, end_offset=21,
    )

    class _Result:
        def __init__(self, rows):
            self.rows = rows

        async def data(self):
            return self.rows

    class _Provider:
        provider = "openai"
        model = "eval-test-model"
        dimensions = 2

        async def embed(self, texts, *, input_type):
            return [[0.0, 1.0] if "unanswerable" in text else [1.0, 0.0]
                    for text in texts]

    class _Db:
        def get_session(self):
            return _Session(self)

    class _Session:
        def __init__(self, db):
            self.db = db

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def run(self, query, **params):
            if "SHOW INDEXES" in query:
                return _Result([
                    {"name": name,
                     "type": "VECTOR" if name.startswith("chunk_embedding_") else "FULLTEXT",
                     "state": "ONLINE"}
                    for name in params["names"]
                ])
            if "FULLTEXT INDEX" in query:
                is_answerable = "unanswerable" not in params["search_query"]
                channel_score = 0.6
            elif "VECTOR INDEX" in query:
                is_answerable = params["query_vector"] == [1.0, 0.0]
                channel_score = 0.8
            else:
                return _Result([])
            if not is_answerable:
                return _Result([])
            return _Result([{
                "chunk_id": "gold-chunk", "document_sha256": DOC,
                "raw_text": "Raw source evidence.",
                "sources_json": [{
                    "page_number": source.page_number, "block_id": source.block_id,
                    "block_reading_order": source.block_reading_order,
                    "block_type": source.block_type,
                    "source_text_length": source.source_text_length,
                    "start_offset": source.start_offset, "end_offset": source.end_offset,
                }],
                "ingestion_run_id": "run-1", "score": channel_score,
            }])

    class _AnswerEvaluator:
        async def evaluate(self, example, answer, evidence):
            return answer == "externally supplied answer" and bool(evidence)

    reference = EvaluationEvidenceReference(
        document_sha256=DOC, cited_page=1, source_page_numbers=(1,),
        quote="Raw source evidence.", chunk_ids=("gold-chunk",),
    )
    dataset = FrozenEvaluationDataset(
        dataset_id="three-strategy-evaluation", version="v1", synthetic=False,
        annotation_source="unit test", corpus_snapshot={DOC: "run-1"},
        queries=(
            {"query_id": "known", "query": "answerable query",
             "relevant_chunk_ids": ["gold-chunk"], "expected_document_ids": [DOC],
             "unanswerable": False, "evidence_references": [reference]},
            {"query_id": "unknown", "query": "unanswerable query",
             "relevant_chunk_ids": [], "expected_document_ids": [],
             "unanswerable": True},
        ),
    )
    db = _Db()
    config = EmbeddingConfig("openai", _Provider.model, 2)
    fulltext = FullTextRetriever(db, candidate_multiplier=3)
    vector = VectorRetriever(db, _Provider(), config, candidate_multiplier=4)
    retrievers = {
        "fulltext": fulltext,
        "vector": vector,
        "hybrid": HybridRetriever(vector, fulltext),
    }

    async def run():
        for strategy, retriever in retrievers.items():
            result, artifact = await run_and_save_evaluation(
                dataset, retriever, artifact_directory=tmp_path / strategy,
                abstention_score_threshold=0.0,
                answers={"known": "externally supplied answer"},
                answer_evaluator=_AnswerEvaluator(),
                citations={"known": [AnswerCitation(
                    document_sha256=DOC, page=1, quote="Raw source evidence.",
                    chunk_id="gold-chunk",
                )]},
            )
            assert result["configuration"]["strategy"] == strategy
            assert result["configuration"]["retrieval"]["strategy"] == strategy
            assert result["per_query"][0]["evidence"][0]["retrieval_channels"]
            assert result["per_query"][0]["abstention_score_type"] == {
                "fulltext": "fulltext_relevance_score",
                "vector": "vector_relevance_score",
                "hybrid": "rrf_score",
            }[strategy]
            assert result["per_query"][0]["abstained"] is False
            assert result["per_query"][1]["abstained"] is True
            assert result["metrics"]["citation_grounding_rate"] == 1.0
            assert result["metrics"]["citation_evaluation_coverage"] == 1.0
            assert result["metrics"]["answerable_queries_with_answer_evaluation"] == 1
            assert result["metrics"]["answerable_answer_evaluation_coverage"] == 1.0
            assert artifact.exists()

    asyncio.run(run())
