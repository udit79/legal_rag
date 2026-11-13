"""Unit tests for the embedding-index execution script; all Neo4j calls are mocked."""

import asyncio

import pytest

from src.graph.embeddings import EmbeddingConfig
from scripts import index_neo4j_embeddings as runner


def _config():
    return EmbeddingConfig("openai", "text-embedding-3-small", 1024)


def _index_row(config, *, state="ONLINE", percent=100.0, **overrides):
    row = {
        "name": config.vector_index_name,
        "state": state,
        "populationPercent": percent,
        "options": {"indexConfig": {
            "vector.dimensions": 1024,
            "vector.similarity_function": "cosine",
        }},
        "labelsOrTypes": ["CHUNK"],
        "properties": sorted(runner.REQUIRED_INDEX_PROPERTIES),
        "failureMessage": None,
    }
    row.update(overrides)
    return row


class _Result:
    def __init__(self, rows=(), one=None):
        self._rows = list(rows)
        self._one = one

    async def data(self):
        return self._rows

    async def single(self):
        return self._one


class _Session:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def run(self, query, **params):
        self.db.queries.append((query, params))
        if query == runner._CURRENT_ELIGIBLE_COUNT_QUERY:
            return _Result(one={"count": self.db.chunk_count})
        if query == runner._SHOW_VECTOR_INDEX_QUERY:
            rows = self.db.index_rows
            return _Result([row for row in rows if row["name"] == params["index_name"]])
        if query == runner._CURRENT_EMBEDDING_VERIFICATION_QUERY:
            return _Result(self.db.embedding_rows)
        raise AssertionError(f"Unexpected Neo4j query: {query}")


class _Db:
    database = "rag"

    def __init__(self, config, *, count=2, index_rows=None, embedding_rows=None):
        self.chunk_count = count
        self.index_rows = index_rows if index_rows is not None else [_index_row(config)]
        self.embedding_rows = embedding_rows if embedding_rows is not None else [
            {"chunk_id": f"c{index}", "embedding_space_id": config.space_id,
             "embedding_model": config.model, "embedding_dimensions": 1024,
             "vector_length": 1024}
            for index in range(count)
        ]
        self.queries = []
        self.verify_calls = 0
        self.close_calls = 0

    def get_session(self):
        return _Session(self)

    async def verify_connection(self):
        self.verify_calls += 1

    async def close(self):
        self.close_calls += 1


class _Provider:
    provider = "openai"
    model = "text-embedding-3-small"
    dimensions = 1024

    def __init__(self):
        self.calls = []

    async def embed(self, texts, *, input_type):
        self.calls.append((list(texts), input_type))
        return [[0.25] * 1024 for _ in texts]


def test_validate_config_requires_expected_provider_model_dimensions_and_database():
    runner.validate_config(_config(), "rag")
    with pytest.raises(ValueError, match="provider"):
        runner.validate_config(EmbeddingConfig("voyage", "text-embedding-3-small", 1024), "rag")
    with pytest.raises(ValueError, match="model"):
        runner.validate_config(EmbeddingConfig("openai", "other", 1024), "rag")
    with pytest.raises(ValueError, match="dimensions"):
        runner.validate_config(EmbeddingConfig("openai", "text-embedding-3-small", 768), "rag")
    with pytest.raises(ValueError, match="expected 'rag'"):
        runner.validate_config(_config(), "neo4j")


def test_vector_index_validation_accepts_neo4j_uppercase_cosine_value():
    config = _config()
    row = _index_row(config)
    row["options"]["indexConfig"]["vector.similarity_function"] = "COSINE"
    runner.validate_vector_index_definition(row, config)


@pytest.mark.parametrize("change", [
    {"labelsOrTypes": ["OTHER"]},
    {"properties": ["embedding"]},
    {"options": {"indexConfig": {"vector.dimensions": 1536,
                                  "vector.similarity_function": "cosine"}}},
    {"options": {"indexConfig": {"vector.dimensions": 1024,
                                  "vector.similarity_function": "euclidean"}}},
])
def test_existing_vector_index_with_incompatible_definition_fails_closed(change):
    row = _index_row(_config(), **change)
    with pytest.raises(ValueError, match="incompatible|lacks required"):
        runner.validate_vector_index_definition(row, _config())


def test_embedding_index_run_verifies_api_and_graph_and_is_idempotent():
    async def run():
        config = _config()
        provider = _Provider()
        dbs = []
        writes = []
        indexed_results = iter((2, 0))

        def db_factory():
            db = _Db(config)
            dbs.append(db)
            return db

        async def indexer(db, selected_provider, selected_config):
            assert selected_provider is provider
            assert selected_config == config
            writes.append("index_embeddings")
            return next(indexed_results)

        async def index_creator(db, selected_config):
            assert selected_config == config
            writes.append("create_index")

        results = []
        for _ in range(2):
            results.append(await runner._execute(
                config_loader=lambda: config,
                provider_factory=lambda selected_config: provider,
                db_factory=db_factory,
                indexer=indexer,
                index_creator=index_creator,
                expected_chunks=2,
            ))

        assert results[0]["newly_indexed_chunks"] == 2
        assert results[0]["already_indexed_chunks"] == 0
        assert results[1]["newly_indexed_chunks"] == 0
        assert results[1]["already_indexed_chunks"] == 2
        assert all(result["verified_current_embeddings"] == 2 for result in results)
        assert all(result["vector_index_state"] == "ONLINE" for result in results)
        assert provider.calls == [([runner.SMOKE_TEXT], "document")] * 2
        assert writes == ["index_embeddings", "create_index"] * 2
        assert all(db.verify_calls == 1 and db.close_calls == 1 for db in dbs)

    asyncio.run(run())


def test_embedding_index_run_refuses_wrong_database_and_still_closes_driver():
    class _WrongDb(_Db):
        database = "neo4j"

    async def run():
        config = _config()
        db = _WrongDb(config)
        with pytest.raises(ValueError, match="expected 'rag'"):
            await runner._execute(
                config_loader=lambda: config,
                provider_factory=lambda _: _Provider(),
                db_factory=lambda: db,
                expected_chunks=2,
            )
        assert db.close_calls == 1

    asyncio.run(run())


def test_failed_existing_index_stops_before_embedding_api_or_writes():
    async def run():
        config = _config()
        provider = _Provider()
        db = _Db(config, index_rows=[_index_row(
            config, state="FAILED", percent=0.0, failureMessage="bad vector dimensions"
        )])
        indexed = []
        with pytest.raises(RuntimeError, match="FAILED.*bad vector dimensions"):
            await runner._execute(
                config_loader=lambda: config,
                provider_factory=lambda _: provider,
                db_factory=lambda: db,
                indexer=lambda *_: indexed.append(True),
                expected_chunks=2,
            )
        assert not provider.calls
        assert not indexed
        assert db.close_calls == 1

    asyncio.run(run())


def test_unexpected_current_chunk_count_stops_before_embedding_api_or_writes():
    async def run():
        config = _config()
        provider = _Provider()
        db = _Db(config, count=1)
        with pytest.raises(RuntimeError, match="Expected 2 current eligible chunks.*found 1"):
            await runner._execute(
                config_loader=lambda: config,
                provider_factory=lambda _: provider,
                db_factory=lambda: db,
                expected_chunks=2,
            )
        assert not provider.calls
        assert db.close_calls == 1

    asyncio.run(run())


@pytest.mark.parametrize("change", [
    {"embedding_space_id": "wrong-space"},
    {"embedding_model": "other-model"},
    {"embedding_dimensions": 768},
    {"vector_length": None},
    {"vector_length": 768},
])
def test_final_vector_verification_rejects_missing_or_mismatched_metadata(change):
    async def run():
        config = _config()
        row = {
            "chunk_id": "bad-chunk", "embedding_space_id": config.space_id,
            "embedding_model": config.model, "embedding_dimensions": config.dimensions,
            "vector_length": config.dimensions,
        }
        row.update(change)
        db = _Db(config, count=1, embedding_rows=[row])
        with pytest.raises(RuntimeError, match="bad-chunk.*embedding metadata or vector length"):
            await runner._verify_all_current_embeddings(db, config, expected_chunks=1)

    asyncio.run(run())


def test_smoke_vector_validation_rejects_non_finite_values(monkeypatch):
    config = _config()
    db = _Db(config)
    provider = _Provider()

    async def invalid_embed(texts, *, input_type):
        return [[float("nan")] * 1024]

    provider.embed = invalid_embed

    async def run():
        # embed_texts rejects the bad value; execution adds its API phase context.
        with pytest.raises(RuntimeError, match="smoke test failed"):
            await runner._execute(
                config_loader=lambda: config,
                provider_factory=lambda _: provider,
                db_factory=lambda: db,
                indexer=lambda *_: asyncio.sleep(0, result=0),
                index_creator=lambda *_: asyncio.sleep(0),
                expected_chunks=2,
            )
        assert db.close_calls == 1

    asyncio.run(run())


def test_vector_index_wait_requires_online_full_population_and_reports_timeout():
    async def run():
        config = _config()
        populating = _Db(config, index_rows=[_index_row(config, state="POPULATING", percent=42.0)])
        now = [0.0]

        async def sleep(seconds):
            now[0] += seconds

        with pytest.raises(TimeoutError, match="POPULATING.*42.0"):
            await runner.wait_for_vector_index(
                populating, config, timeout_seconds=3.0, poll_interval_seconds=1.0,
                clock=lambda: now[0], sleep=sleep,
            )

    asyncio.run(run())


@pytest.mark.parametrize("timeout, interval", [
    (0, 1), (-1, 1), (float("inf"), 1), (10, 0), (10, float("nan")),
])
def test_index_wait_settings_must_be_finite_and_positive(timeout, interval):
    with pytest.raises(ValueError, match="finite and positive"):
        runner._validate_wait_settings(timeout, interval)


def test_safe_error_output_redacts_configured_and_common_api_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-key")
    message = runner.safe_error_message(RuntimeError(
        "provider rejected secret-openai-key and sk-proj-abcdefghijklmno"
    ))
    assert "secret-openai-key" not in message
    assert "sk-proj-abcdefghijklmno" not in message
    assert "[REDACTED]" in message
