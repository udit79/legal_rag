import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.graph.embeddings import (
    EmbeddingConfig,
    create_embedding_provider,
    embed_texts,
    index_current_chunks,
    vector_index_statement,
)
from src.graph.evaluation import (
    AnswerCitation,
    EvaluationEvidenceReference,
    FrozenEvaluationDataset,
    load_evaluation_dataset,
    run_and_save_evaluation,
    run_evaluation,
)
from src.graph.retrieval import (
    EvidenceBundle,
    FullTextRetriever,
    HybridRetriever,
    RetrievalHit,
    RRFConfig,
    SearchIndexUnavailable,
    VectorRetriever,
    reciprocal_rank_fusion,
)
from src.ingestion.chunk_contracts import ChunkSource

FIXTURE = Path("src/graph/evaluation/dev_v1.json")
CORPUS_FIXTURE = Path("src/graph/evaluation/legal_corpus_dev_v1.json")
DOC = "a" * 64


class _Provider:
    provider = "openai"
    model = "test-embedding-model"
    dimensions = 2

    async def embed(self, texts, *, input_type):
        self.input_type = input_type
        return [[1, 0] for _ in texts]


class _Result:
    def __init__(self, records):
        self.records = records

    async def data(self):
        return self.records


class _Session:
    def __init__(self, tx):
        self.tx = tx

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def run(self, query, **params):
        self.tx.calls.append((query, params))
        if "SHOW INDEXES" in query:
            return _Result([{"name": name, "type": "VECTOR" if name != "chunk_retrieval_text_fulltext" else "FULLTEXT",
                             "state": "ONLINE"} for name in params["names"]])
        if "FULLTEXT INDEX" in query:
            return _Result([_record("c1", DOC, "Raw source quote", 2.2)])
        if "VECTOR INDEX" in query:
            return _Result([_record("c1", DOC, "Raw source quote", 0.91)])
        return _Result([])


class _Db:
    def __init__(self):
        self.calls = []

    def get_session(self):
        return _Session(self)


class _NotReadyDb(_Db):
    def __init__(self):
        super().__init__()

    def get_session(self):
        class _SessionNotReady(_Session):
            async def run(self, query, **params):
                self.tx.calls.append((query, params))
                if "SHOW INDEXES" in query:
                    return _Result([{"name": "chunk_retrieval_text_fulltext",
                                     "type": "FULLTEXT", "state": "POPULATING"}])
                return _Result([])
        return _SessionNotReady(self)


class _IndexResult(_Result):
    def __init__(self, records=(), single=None):
        super().__init__(records)
        self.single_record = single

    async def single(self):
        return self.single_record

    async def consume(self):
        return None


class _IndexTx:
    def __init__(self, conflicting_spaces=0, incomplete_embeddings=0, current_rows=None):
        self.conflicting_spaces = conflicting_spaces
        self.incomplete_embeddings = incomplete_embeddings
        self.stored_rows = None
        self.current_rows = current_rows or [{
            "chunk_id": "current-chunk", "retrieval_text": "normalized search text",
            "embedding": None, "embedding_space_id": None, "embedding_model": None,
            "embedding_dimensions": None, "source_sha256": DOC,
            "chunking_policy_version": "policy-v1", "ingestion_run_id": "run-current",
        }]

    async def run(self, query, **params):
        if "RETURN DISTINCT c.id AS chunk_id" in query:
            return _IndexResult(self.current_rows)
        if "RETURN g.embedding_space_id" in query:
            return _IndexResult(single={"embedding_space_id": None})
        if "WHERE (c.embedding IS NULL" in query:
            return _IndexResult([{"count": self.incomplete_embeddings}])
        if "c.embedding IS NOT NULL" in query:
            return _IndexResult([{"count": self.conflicting_spaces}])
        if "UNWIND $rows AS row" in query:
            self.stored_rows = params["rows"]
            return _IndexResult(single={"stored": len(self.stored_rows)})
        return _IndexResult()


class _IndexSession(_Session):
    async def run(self, query, **params):
        if "RETURN DISTINCT c.id AS chunk_id" in query:
            return _IndexResult([{
                "chunk_id": "current-chunk", "retrieval_text": "normalized search text",
                "embedding": None, "embedding_space_id": None,
                "embedding_model": None, "embedding_dimensions": None,
                "source_sha256": DOC,
                "chunking_policy_version": "policy-v1",
                "ingestion_run_id": "run-current",
            }])
        return _IndexResult()

    async def execute_write(self, callback):
        return await callback(self.tx)


class _IndexDb:
    def __init__(self, tx):
        self.tx = tx

    def get_session(self):
        return _IndexSession(self.tx)


def _record(chunk_id, document_id, raw_text, score):
    return {
        "chunk_id": chunk_id,
        "document_sha256": document_id,
        "raw_text": raw_text,
        "sources_json": json.dumps([{
            "page_number": 2, "block_id": "page-2-block-4",
            "block_reading_order": 4, "block_type": "TEXT",
            "source_text_length": len(raw_text), "start_offset": 0,
            "end_offset": len(raw_text),
        }]),
        "ingestion_run_id": "run-current",
        "score": score,
    }


def _hit(chunk_id="same", channel="vector", doc=DOC, raw="Original raw text", score=0.8):
    source = ChunkSource(
        page_number=1, block_id="page-1-block-0", block_reading_order=0,
        block_type="TEXT", source_text_length=len(raw), start_offset=0,
        end_offset=len(raw),
    )
    return RetrievalHit(chunk_id, doc, raw, (source,), "run-1", channel, score)


def test_embedding_space_requires_provider_model_dimensions_and_checks_output():
    config = EmbeddingConfig("openai", "test-embedding-model", 2)

    async def run():
        vectors = await embed_texts(_Provider(), config, ["raw retrieval query"], input_type="query")
        assert vectors == [[1.0, 0.0]]
        assert _Provider.model == config.model
        with pytest.raises(ValueError, match="provider does not match"):
            await embed_texts(_Provider(), EmbeddingConfig("voyage", "test-embedding-model", 2),
                              ["query"], input_type="query")
    asyncio.run(run())


def test_embedding_configuration_has_no_implicit_provider_or_model(monkeypatch):
    for key in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="Set EMBEDDING_PROVIDER"):
        EmbeddingConfig.from_env()
    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("EMBEDDING_MODEL", "explicit-model")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
    assert EmbeddingConfig.from_env() == EmbeddingConfig("voyage", "explicit-model", 768)


def test_openai_and_voyage_adapters_use_explicit_model_and_dimensions(monkeypatch):
    import openai
    import voyageai

    calls = []

    class _OpenAIClient:
        def __init__(self, api_key):
            assert api_key == "unit-test-key"
            self.embeddings = SimpleNamespace(create=self.create)

        async def create(self, **kwargs):
            calls.append(("openai", kwargs))
            return SimpleNamespace(model=kwargs["model"], data=[SimpleNamespace(embedding=[0.1, 0.2])])

    class _VoyageClient:
        def __init__(self, api_key):
            assert api_key == "unit-test-key"

        async def embed(self, texts, **kwargs):
            calls.append(("voyage", {"texts": texts, **kwargs}))
            return SimpleNamespace(embeddings=[[0.3, 0.4]])

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("VOYAGE_API_KEY", "unit-test-key")
    monkeypatch.setattr(openai, "AsyncOpenAI", _OpenAIClient)
    monkeypatch.setattr(voyageai, "AsyncClient", _VoyageClient)

    async def run():
        openai_config = EmbeddingConfig("openai", "openai-unit-test", 2)
        voyage_config = EmbeddingConfig("voyage", "voyage-unit-test", 2)
        openai_provider = create_embedding_provider(openai_config)
        voyage_provider = create_embedding_provider(voyage_config)
        assert await embed_texts(
            openai_provider, openai_config, ["text"], input_type="document"
        ) == [[0.1, 0.2]]
        assert await embed_texts(
            voyage_provider, voyage_config, ["text"], input_type="query"
        ) == [[0.3, 0.4]]
    asyncio.run(run())
    assert calls[0][1]["model"] == "openai-unit-test"
    assert calls[0][1]["dimensions"] == 2
    assert calls[1][1]["model"] == "voyage-unit-test"
    assert calls[1][1]["output_dimension"] == 2
    assert calls[1][1]["input_type"] == "query"


def test_vector_index_requires_explicit_dimensions_and_uses_fixed_configured_index():
    config = EmbeddingConfig("voyage", "voyage-test", 768)
    statement = vector_index_statement(config)
    assert f"CREATE VECTOR INDEX {config.vector_index_name} IF NOT EXISTS" in statement
    assert "`vector.dimensions`: 768" in statement
    assert "`vector.similarity_function`: 'cosine'" in statement
    assert "WITH [c.is_current, c.source_sha256, c.chunking_policy_version" in statement
    assert "c.embedding_space_id, c.embedding_model, c.embedding_dimensions" in statement


def test_embedding_indexer_embeds_retrieval_text_and_rejects_mixed_spaces():
    async def run():
        config = EmbeddingConfig("openai", _Provider.model, 2)
        provider = _Provider()
        tx = _IndexTx()
        assert await index_current_chunks(_IndexDb(tx), provider, config) == 1
        assert provider.input_type == "document"
        assert tx.stored_rows[0]["embedding_space_id"] == config.space_id
        assert tx.stored_rows[0]["embedding"] == [1.0, 0.0]
        with pytest.raises(ValueError, match="different embedding space"):
            await index_current_chunks(
                _IndexDb(_IndexTx(conflicting_spaces=1)), provider, config
            )
    asyncio.run(run())


def test_embedding_indexing_closes_read_session_before_provider_and_reopens_for_write():
    class _TrackedDb(_IndexDb):
        open_sessions = 0
        session_count = 0

        def get_session(self):
            db = self

            class _TrackedSession(_IndexSession):
                async def __aenter__(self):
                    db.open_sessions += 1
                    db.session_count += 1
                    return self

                async def __aexit__(self, *_):
                    db.open_sessions -= 1

            return _TrackedSession(self.tx)

    class _SessionAwareProvider(_Provider):
        def __init__(self, db):
            self.db = db

        async def embed(self, texts, *, input_type):
            assert self.db.open_sessions == 0
            return await super().embed(texts, input_type=input_type)

    async def run():
        db = _TrackedDb(_IndexTx())
        config = EmbeddingConfig("openai", _Provider.model, 2)
        count = await index_current_chunks(db, _SessionAwareProvider(db), config)
        assert count == 1
        assert db.open_sessions == 0
        assert db.session_count == 2
    asyncio.run(run())


def test_embedding_indexer_revalidates_current_chunk_and_run_snapshot_before_write():
    async def run():
        changed = [{
            "chunk_id": "replacement-chunk", "retrieval_text": "normalized search text",
            "embedding": None, "embedding_space_id": None, "embedding_model": None,
            "embedding_dimensions": None, "source_sha256": DOC,
            "chunking_policy_version": "policy-v2", "ingestion_run_id": "run-newer",
        }]
        tx = _IndexTx(current_rows=changed)
        with pytest.raises(RuntimeError, match="snapshot changed"):
            await index_current_chunks(
                _IndexDb(tx), _Provider(), EmbeddingConfig("openai", _Provider.model, 2)
            )
        assert tx.stored_rows is None
    asyncio.run(run())


def test_embedding_indexer_does_not_overwrite_embedding_completed_by_same_space_writer():
    async def run():
        config = EmbeddingConfig("openai", _Provider.model, 2)
        row = {
            "chunk_id": "current-chunk", "retrieval_text": "normalized search text",
            "embedding": [0.5, 0.5], "embedding_space_id": config.space_id,
            "embedding_model": config.model, "embedding_dimensions": config.dimensions,
            "source_sha256": DOC, "chunking_policy_version": "policy-v1",
            "ingestion_run_id": "run-current",
        }
        tx = _IndexTx(current_rows=[row])
        assert await index_current_chunks(_IndexDb(tx), _Provider(), config) == 0
        assert tx.stored_rows is None
    asyncio.run(run())


@pytest.mark.parametrize("partial", [
    {"embedding": [1.0, 0.0]},
    {"embedding_space_id": "space"},
    {"embedding": [1.0, 0.0], "embedding_space_id": "space",
     "embedding_model": "other-model", "embedding_dimensions": 2},
    {"embedding": [1.0], "embedding_space_id": "space",
     "embedding_model": "test-embedding-model", "embedding_dimensions": 2},
])
def test_embedding_indexer_fails_closed_on_incomplete_or_conflicting_metadata(partial):
    class _PartialSession(_IndexSession):
        async def run(self, query, **params):
            if "RETURN DISTINCT c.id AS chunk_id" in query:
                row = {
                    "chunk_id": "current-chunk", "retrieval_text": "normalized search text",
                    "embedding": None, "embedding_space_id": None, "embedding_model": None,
                    "embedding_dimensions": None, "source_sha256": DOC,
                    "ingestion_run_id": "run-current",
                }
                row.update(partial)
                return _IndexResult([row])
            return _IndexResult()

    class _PartialDb(_IndexDb):
        def get_session(self):
            return _PartialSession(self.tx)

    async def run():
        config = EmbeddingConfig("openai", _Provider.model, 2)
        with pytest.raises(ValueError, match="incomplete or conflicting embedding metadata"):
            await index_current_chunks(_PartialDb(_IndexTx()), _Provider(), config)
    asyncio.run(run())


def test_fulltext_and_vector_retrievers_are_independent_and_return_raw_provenance():
    async def run():
        db = _Db()
        config = EmbeddingConfig("openai", _Provider.model, _Provider.dimensions)
        provider = _Provider()
        fulltext = await FullTextRetriever(db, candidate_multiplier=4).search("retrieval terms", limit=3)
        vector = await VectorRetriever(db, provider, config).search("semantic question", limit=3)
        assert [item.channel for item in fulltext] == ["fulltext"]
        assert [item.channel for item in vector] == ["vector"]
        assert vector[0].raw_text == "Raw source quote"
        assert vector[0].sources[0].block_id == "page-2-block-4"
        assert provider.input_type == "query"
        assert "retrieval_text" not in fulltext[0].__dict__
        for query, _ in db.calls:
            if "SHOW INDEXES" in query:
                continue
            assert "c.is_current = true" in query
            assert "c.chunking_policy_version = d.current_chunking_policy_version" in query
            assert "d.current_ingestion_run_id IS NOT NULL" in query
        fulltext_query, fulltext_params = next((query, params) for query, params in db.calls
                                               if "FULLTEXT INDEX" in query)
        assert "CYPHER 25" in fulltext_query
        assert "FULLTEXT INDEX chunk_retrieval_text_fulltext" in fulltext_query
        assert "$index_name" not in fulltext_query
        assert fulltext_params == {"search_query": "retrieval terms", "candidate_count": 12, "limit": 3}
        assert "LIMIT $limit" in fulltext_query.split("RETURN DISTINCT", 1)[1]
        vector_query, vector_params = next((query, params) for query, params in db.calls
                                           if "VECTOR INDEX" in query)
        assert "db.index.vector.queryNodes" not in vector_query
        assert "WHERE c.is_current = true" in vector_query
        assert "c.embedding_space_id = $embedding_space_id" in vector_query
        assert vector_params["embedding_space_id"] == config.space_id
    asyncio.run(run())


def test_retrieval_uses_search_and_stops_when_indexes_are_not_online():
    async def run():
        with pytest.raises(SearchIndexUnavailable, match="not ONLINE"):
            await FullTextRetriever(_NotReadyDb()).search("query")
    asyncio.run(run())


def test_hybrid_retrieval_checks_both_index_states_before_any_search():
    async def run():
        db = _NotReadyDb()
        config = EmbeddingConfig("openai", _Provider.model, _Provider.dimensions)
        retriever = HybridRetriever(
            VectorRetriever(db, _Provider(), config), FullTextRetriever(db)
        )
        with pytest.raises(SearchIndexUnavailable):
            await retriever.search("query")
        assert not any("VECTOR INDEX" in query or "FULLTEXT INDEX" in query
                       for query, _ in db.calls)
    asyncio.run(run())


def test_hybrid_retrieval_checks_index_readiness_once_before_child_queries():
    async def run():
        db = _Db()
        config = EmbeddingConfig("openai", _Provider.model, _Provider.dimensions)
        retriever = HybridRetriever(
            VectorRetriever(db, _Provider(), config), FullTextRetriever(db)
        )
        await retriever.search("query", limit=2)
        assert sum("SHOW INDEXES" in query for query, _ in db.calls) == 1
    asyncio.run(run())


def test_rrf_combines_channels_and_breaks_score_ties_by_chunk_id():
    fused = reciprocal_rank_fusion(
        [_hit("b", "vector"), _hit("a", "vector")],
        [_hit("a", "fulltext"), _hit("b", "fulltext")],
        RRFConfig(rank_constant=0), limit=2,
    )
    assert [bundle.chunk_id for bundle in fused] == ["a", "b"]
    assert fused[0].retrieval_channels == ("vector", "fulltext")
    assert fused[0].rrf_score == pytest.approx(1.5)
    assert fused[0].raw_text == "Original raw text"
    assert fused[0].sources[0].block_id == "page-1-block-0"
    assert fused[0].model_dump()["vector_score"] == 0.8


def test_rrf_rejects_retrievers_that_disagree_about_source_evidence():
    with pytest.raises(ValueError, match="inconsistent evidence"):
        reciprocal_rank_fusion(
            [_hit("same", "vector", raw="A")],
            [_hit("same", "fulltext", raw="B")],
        )


def test_frozen_dev_evaluation_dataset_and_metrics_are_saved():
    class _Retriever:
        configuration = {"embedding_model": "test", "embedding_dimensions": 2}

        async def search(self, query, limit):
            if "habeas" in query:
                return [EvidenceBundle(
                    chunk_id="e5572d4b0681bb3376829394b13439bbb26f016cc6270395d458aca30fc90794",
                    document_sha256="a9f5c03dcfc0962d0cfef3e2dcab9416f2c2839e78884b2ea32b9e326d05867a",
                    raw_text="Original habeas source text",
                    sources=(_hit().sources[0],), ingestion_run_id="run",
                    retrieval_channels=("fulltext",), fulltext_score=1.0, rrf_score=0.05,
                )]
            if "tax" in query:
                return []
            return []

    dataset = load_evaluation_dataset(FIXTURE)
    assert dataset.synthetic is True

    async def run():
        with TemporaryDirectory(dir=".") as output_dir:
            result, output = await run_and_save_evaluation(
                dataset, _Retriever(), k=3, abstention_rrf_threshold=0.01,
                artifact_directory=output_dir
            )
            assert result["metrics"]["recall_at_3"] == pytest.approx(1 / 3)
            assert result["metrics"]["mrr"] == pytest.approx(1 / 3)
            assert result["metrics"]["retrieved_document_precision"] == pytest.approx(1 / 3)
            assert result["metrics"]["document_recall_at_3"] == pytest.approx(1 / 3)
            assert result["metrics"]["unanswerable_query_abstention"] == 1.0
            assert result["metrics"]["citation_accuracy"] is None
            assert result["metrics"]["answer_correctness"] is None
            assert result["metrics"]["answerable_queries_with_answer_evaluation"] is None
            assert result["metrics"]["answerable_answer_evaluation_coverage"] is None
            saved = json.loads(output.read_text(encoding="utf-8"))
            assert saved["configuration"]["retrieval"]["embedding_model"] == "test"
            assert len(saved["per_query"]) == 4
    asyncio.run(run())


def test_manual_development_fixture_is_pinned_to_ingested_corpus_runs():
    synthetic = load_evaluation_dataset(FIXTURE)
    corpus = load_evaluation_dataset(CORPUS_FIXTURE)
    assert synthetic.synthetic is True
    assert corpus.synthetic is False
    assert corpus.annotation_source == "test_data/evaluation.jsonl (human-adjudicated dev split)"
    assert len(corpus.queries) == 27
    assert len(corpus.corpus_snapshot) == 6
    assert sum(len(query.evidence_references) for query in corpus.queries) == 65
    assert all(reference.chunk_ids for query in corpus.queries
               for reference in query.evidence_references)
    assert all(reference.cited_page in reference.source_page_numbers
               for query in corpus.queries for reference in query.evidence_references)
    assert any(reference.annotated_page != reference.cited_page
               for query in corpus.queries for reference in query.evidence_references)
    with pytest.raises(TypeError):
        corpus.corpus_snapshot[next(iter(corpus.corpus_snapshot))] = "changed"
    with pytest.raises((AttributeError, TypeError)):
        corpus.corpus_snapshot.update({"b" * 64: "changed"})
    backing = {"a" * 64: "run-original"}
    immutable = type(corpus.corpus_snapshot)(backing)
    backing["a" * 64] = "mutated-source"
    assert immutable["a" * 64] == "run-original"
    exposed_backing = immutable._ImmutableMapping__data
    with pytest.raises(TypeError):
        exposed_backing["a" * 64] = "mutated-through-reference"
    with pytest.raises(ValidationError):
        corpus.corpus_snapshot = {}
    dumped = corpus.model_dump()
    assert isinstance(dumped["corpus_snapshot"], dict)
    assert dumped["corpus_snapshot"] == dict(corpus.corpus_snapshot)
    assert FrozenEvaluationDataset.model_validate_json(corpus.model_dump_json()) == corpus
    assert isinstance(
        FrozenEvaluationDataset.model_validate_json(corpus.model_dump_json()).corpus_snapshot,
        type(corpus.corpus_snapshot),
    )


def test_document_metrics_exclude_unanswerable_and_citation_accuracy_checks_evidence():
    dataset = FrozenEvaluationDataset(
        dataset_id="metric-test", version="1", synthetic=True,
        queries=(
            {"query_id": "answerable", "query": "known", "relevant_chunk_ids": ["gold"],
             "expected_document_ids": [DOC], "unanswerable": False},
            {"query_id": "unknown", "query": "unknown", "relevant_chunk_ids": [],
             "expected_document_ids": [], "unanswerable": True},
        ),
    )

    class _Retriever:
        async def search(self, query, limit):
            if query == "known":
                return [EvidenceBundle(
                    chunk_id="gold", document_sha256=DOC, raw_text="source",
                    sources=(_hit().sources[0],), ingestion_run_id="run",
                    retrieval_channels=("fulltext",), fulltext_score=0.5, rrf_score=0.5,
                )]
            return [EvidenceBundle(
                chunk_id="false-positive", document_sha256="b" * 64, raw_text="source",
                sources=(_hit().sources[0],), ingestion_run_id="run",
                retrieval_channels=("fulltext",), fulltext_score=0.1, rrf_score=0.1,
            )]

    async def run():
        result = await run_evaluation(
            dataset, _Retriever(), k=2, abstention_rrf_threshold=0.2,
        )
        assert result["metrics"]["retrieved_document_precision"] == 1.0
        assert result["metrics"]["document_recall_at_2"] == 1.0
        assert result["metrics"]["unanswerable_query_abstention"] == 1.0
        assert result["metrics"]["citation_accuracy"] is None
        assert result["per_query"][1]["retrieved_document_precision"] is None
        assert result["per_query"][1]["document_recall_at_k"] is None

    asyncio.run(run())
    async def invalid_threshold():
        await run_evaluation(dataset, _Retriever(), abstention_rrf_threshold=float("nan"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        asyncio.run(invalid_threshold())


@pytest.mark.parametrize("overrides, expected_grounding", [
    ({}, 1.0),
    ({"document_sha256": "b" * 64}, 0.0),
    ({"page": 2}, 1.0),
    ({"quote": "unsupported words"}, 1.0),
    ({"chunk_id": "not-retrieved"}, 0.0),
])
def test_citation_accuracy_validates_document_page_quote_and_chunk(overrides, expected_grounding):
    reference = EvaluationEvidenceReference(
        document_sha256=DOC, cited_page=1, source_page_numbers=(1,),
        quote="Exact citation phrase is supported here.", chunk_ids=("gold",),
    )
    dataset = FrozenEvaluationDataset(
        dataset_id="citation-test", version="1", synthetic=True,
        queries=({"query_id": "q", "query": "question", "relevant_chunk_ids": ["gold"],
                  "expected_document_ids": [DOC], "unanswerable": False,
                  "evidence_references": [reference.model_dump()]},),
    )

    class _Retriever:
        async def search(self, query, limit):
            return [EvidenceBundle(
                chunk_id="gold", document_sha256=DOC,
                raw_text="Prefix. Exact citation phrase is supported here. Suffix.",
                sources=(_hit().sources[0],), ingestion_run_id="run",
                retrieval_channels=("fulltext",), fulltext_score=1.0, rrf_score=1.0,
            )]

    citation = {"document_sha256": DOC, "page": 1,
                "quote": "Exact citation phrase is supported here.", "chunk_id": "gold"}
    citation.update(overrides)

    async def run():
        result = await run_evaluation(
            dataset, _Retriever(), abstention_rrf_threshold=0.1,
            citations={"q": [AnswerCitation.model_validate(citation)]},
        )
        assert result["metrics"]["citation_grounding_rate"] == expected_grounding
        assert result["metrics"]["citation_accuracy"] == (1.0 if not overrides else 0.0)
    asyncio.run(run())


def test_citation_accuracy_requires_actual_chunk_source_page_provenance():
    reference = EvaluationEvidenceReference(
        document_sha256=DOC, cited_page=1, source_page_numbers=(1,),
        quote="Exact citation phrase is supported here.", chunk_ids=("gold",),
    )
    dataset = FrozenEvaluationDataset(
        dataset_id="citation-page-test", version="1", synthetic=True,
        queries=({"query_id": "q", "query": "question", "relevant_chunk_ids": ["gold"],
                  "expected_document_ids": [DOC], "unanswerable": False,
                  "evidence_references": [reference.model_dump()]},),
    )

    class _Retriever:
        async def search(self, query, limit):
            source = ChunkSource(
                page_number=2, block_id="page-2-block-0", block_reading_order=0,
                block_type="TEXT", source_text_length=48, start_offset=0, end_offset=48,
            )
            return [EvidenceBundle(
                chunk_id="gold", document_sha256=DOC,
                raw_text="Prefix. Exact citation phrase is supported here. Suffix.",
                sources=(source,), ingestion_run_id="run", retrieval_channels=("fulltext",),
                fulltext_score=1.0, rrf_score=1.0,
            )]

    async def run():
        result = await run_evaluation(
            dataset, _Retriever(), abstention_rrf_threshold=0.1,
            citations={"q": [{"document_sha256": DOC, "page": 1,
                               "quote": "Exact citation phrase is supported here.",
                               "chunk_id": "gold"}]},
        )
        assert result["metrics"]["citation_grounding_rate"] == 1.0
        assert result["metrics"]["citation_accuracy"] == 0.0
    asyncio.run(run())


def test_citation_evaluation_counts_missing_answerable_citations_and_reports_coverage():
    reference = EvaluationEvidenceReference(
        document_sha256=DOC, cited_page=1, source_page_numbers=(1,),
        quote="Exact citation phrase is supported here.", chunk_ids=("gold",),
    )
    query_labels = {
        "relevant_chunk_ids": ["gold"], "expected_document_ids": [DOC],
        "unanswerable": False, "evidence_references": [reference.model_dump()],
    }
    dataset = FrozenEvaluationDataset(
        dataset_id="citation-coverage-test", version="1", synthetic=True,
        queries=(
            {"query_id": "missing", "query": "missing citations", **query_labels},
            {"query_id": "provided", "query": "provided citations", **query_labels},
            {"query_id": "empty", "query": "empty citations", **query_labels},
        ),
    )

    class _Retriever:
        async def search(self, query, limit):
            return [EvidenceBundle(
                chunk_id="gold", document_sha256=DOC,
                raw_text="Prefix. Exact citation phrase is supported here. Suffix.",
                sources=(_hit().sources[0],), ingestion_run_id="run",
                retrieval_channels=("fulltext",), fulltext_score=1.0, rrf_score=1.0,
            )]

    async def run():
        result = await run_evaluation(
            dataset, _Retriever(), abstention_rrf_threshold=0.1,
            citations={"provided": [{
                "document_sha256": DOC, "page": 1,
                "quote": "Exact citation phrase is supported here.", "chunk_id": "gold",
            }], "empty": []},
        )
        assert result["metrics"]["citation_evaluation_coverage"] == pytest.approx(2 / 3)
        assert result["metrics"]["citation_accuracy"] == pytest.approx(1 / 3)
        assert result["metrics"]["citation_grounding_rate"] == pytest.approx(1 / 3)
        assert result["per_query"][0]["citation_evaluation_status"] == "MISSING_CITATIONS"
        assert result["per_query"][0]["citation_accuracy"] == 0.0
        assert result["per_query"][1]["citation_evaluation_status"] == "EVALUATED"
        assert result["per_query"][2]["citation_evaluation_status"] == "EVALUATED"
        assert result["per_query"][2]["citation_accuracy"] == 0.0
    asyncio.run(run())


def test_citation_page_must_match_canonical_cited_page():
    reference = EvaluationEvidenceReference(
        document_sha256=DOC, cited_page=1, source_page_numbers=(1, 2),
        quote="Exact citation phrase is supported here.", chunk_ids=("gold",),
    )
    dataset = FrozenEvaluationDataset(
        dataset_id="citation-canonical-page-test", version="1", synthetic=True,
        queries=({"query_id": "q", "query": "question", "relevant_chunk_ids": ["gold"],
                  "expected_document_ids": [DOC], "unanswerable": False,
                  "evidence_references": [reference.model_dump()]},),
    )

    class _Retriever:
        async def search(self, query, limit):
            source = ChunkSource(
                page_number=2, block_id="page-2-block-0", block_reading_order=0,
                block_type="TEXT", source_text_length=48, start_offset=0, end_offset=48,
            )
            return [EvidenceBundle(
                chunk_id="gold", document_sha256=DOC,
                raw_text="Prefix. Exact citation phrase is supported here. Suffix.",
                sources=(source,), ingestion_run_id="run", retrieval_channels=("fulltext",),
                fulltext_score=1.0, rrf_score=1.0,
            )]

    async def run():
        result = await run_evaluation(
            dataset, _Retriever(), abstention_rrf_threshold=0.1,
            citations={"q": [{"document_sha256": DOC, "page": 2,
                               "quote": "Exact citation phrase is supported here.",
                               "chunk_id": "gold"}]},
        )
        assert result["metrics"]["citation_grounding_rate"] == 1.0
        assert result["metrics"]["citation_accuracy"] == 0.0
    asyncio.run(run())


def test_evidence_reference_cited_page_must_be_a_source_page():
    with pytest.raises(ValueError, match="cited page must belong"):
        EvaluationEvidenceReference(
            document_sha256=DOC, cited_page=2, source_page_numbers=(1,),
            quote="quote", chunk_ids=("chunk",),
        )


def test_evaluation_requires_explicit_abstention_threshold_and_scores_abstention():
    dataset = FrozenEvaluationDataset(
        dataset_id="abstention-test", version="1", synthetic=True,
        queries=({"query_id": "unknown", "query": "unknown", "relevant_chunk_ids": [],
                  "expected_document_ids": [], "unanswerable": True},),
    )

    class _Retriever:
        def __init__(self, score):
            self.score = score

        async def search(self, query, limit):
            if self.score is None:
                return []
            return [EvidenceBundle(
                chunk_id="candidate", document_sha256=DOC, raw_text="unrelated",
                sources=(_hit().sources[0],), ingestion_run_id="run",
                retrieval_channels=("fulltext",), fulltext_score=self.score,
                rrf_score=self.score,
            )]

    with pytest.raises(TypeError, match="abstention_rrf_threshold"):
        run_evaluation(dataset, _Retriever(None))

    async def run():
        empty = await run_evaluation(dataset, _Retriever(None), abstention_rrf_threshold=0.4)
        below = await run_evaluation(dataset, _Retriever(0.3), abstention_rrf_threshold=0.4)
        above = await run_evaluation(dataset, _Retriever(0.5), abstention_rrf_threshold=0.4)
        assert empty["per_query"][0]["abstained"] is True
        assert below["per_query"][0]["abstained"] is True
        assert above["per_query"][0]["abstained"] is False
        assert above["configuration"]["abstention_rrf_threshold"] == 0.4
    asyncio.run(run())


def test_answer_correctness_is_an_injected_interface_not_answer_generation():
    dataset = FrozenEvaluationDataset(
        dataset_id="test", version="1", synthetic=True,
        queries=({"query_id": "q", "query": "question", "relevant_chunk_ids": ["x"],
                  "expected_document_ids": [DOC], "unanswerable": False},),
    )

    class _Retriever:
        async def search(self, query, limit):
            return []

    class _Checker:
        async def evaluate(self, example, answer, evidence):
            return answer == "external answer" and not evidence

    async def run():
        result = await run_evaluation(dataset, _Retriever(), abstention_rrf_threshold=0.01,
                                      answers={"q": "external answer"},
                                      answer_evaluator=_Checker())
        assert result["metrics"]["answer_correctness"] == 1.0
        assert result["metrics"]["answerable_queries_with_answer_evaluation"] == 1
        assert result["metrics"]["answerable_answer_evaluation_coverage"] == 1.0
        assert result["per_query"][0]["answer_evaluation_status"] == "EVALUATED"
        with pytest.raises(ValueError, match="unknown query IDs"):
            await run_evaluation(
                dataset, _Retriever(), abstention_rrf_threshold=0.01,
                answers={"unknown": "answer"}, answer_evaluator=_Checker(),
            )
    asyncio.run(run())


def test_answer_evaluation_coverage_reports_missing_answerable_ids():
    dataset = FrozenEvaluationDataset(
        dataset_id="answer-coverage", version="1", synthetic=True,
        queries=(
            {"query_id": "answered", "query": "one", "relevant_chunk_ids": ["x"],
             "expected_document_ids": [DOC], "unanswerable": False},
            {"query_id": "missing", "query": "two", "relevant_chunk_ids": ["y"],
             "expected_document_ids": [DOC], "unanswerable": False},
        ),
    )

    class _Retriever:
        async def search(self, query, limit):
            return []

    class _Checker:
        async def evaluate(self, example, answer, evidence):
            return bool(answer)

    async def run():
        result = await run_evaluation(
            dataset, _Retriever(), abstention_rrf_threshold=0.1,
            answers={"answered": "answer"}, answer_evaluator=_Checker(),
        )
        assert result["metrics"]["answerable_queries_with_answer_evaluation"] == 1
        assert result["metrics"]["answerable_answer_evaluation_coverage"] == 0.5
        assert [row["answer_evaluation_status"] for row in result["per_query"]] == [
            "EVALUATED", "MISSING_ANSWER",
        ]
    asyncio.run(run())
